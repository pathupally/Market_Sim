#include "marketforge/model/loaded_weights.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "marketforge/core/mapped_file.hpp"
#include "marketforge/core/shape.hpp"

namespace marketforge {
namespace {

struct ExpectedTensor {
  std::string name;
  Shape shape;
};

Result<Shape> shape_of(std::initializer_list<std::uint64_t> extents) {
  return make_shape(
      std::span<const std::uint64_t>(extents.begin(), extents.size()));
}

Result<std::vector<ExpectedTensor>>
smollm2_expected_tensors(const ModelSpec& spec) {
  std::vector<ExpectedTensor> expected;
  expected.reserve(static_cast<std::size_t>(spec.layers) * 9U + 2U);

  const auto embedding = shape_of({spec.vocabulary_size, spec.hidden_size});
  const auto norm = shape_of({spec.hidden_size});
  const auto query = shape_of({spec.hidden_size, spec.hidden_size});
  const auto kv_width =
      static_cast<std::uint64_t>(spec.kv_heads) * spec.head_dim;
  const auto key_value = shape_of({kv_width, spec.hidden_size});
  const auto gate_up = shape_of({spec.intermediate_size, spec.hidden_size});
  const auto down = shape_of({spec.hidden_size, spec.intermediate_size});
  if (!embedding || !norm || !query || !key_value || !gate_up || !down) {
    return Result<std::vector<ExpectedTensor>>::failure(
        Status::failure(ErrorCode::invalid_model));
  }

  expected.push_back(
      ExpectedTensor{"model.embed_tokens.weight", embedding.value()});
  for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    expected.push_back(
        ExpectedTensor{prefix + "input_layernorm.weight", norm.value()});
    expected.push_back(
        ExpectedTensor{prefix + "self_attn.q_proj.weight", query.value()});
    expected.push_back(
        ExpectedTensor{prefix + "self_attn.k_proj.weight", key_value.value()});
    expected.push_back(
        ExpectedTensor{prefix + "self_attn.v_proj.weight", key_value.value()});
    expected.push_back(
        ExpectedTensor{prefix + "self_attn.o_proj.weight", query.value()});
    expected.push_back(ExpectedTensor{
        prefix + "post_attention_layernorm.weight", norm.value()});
    expected.push_back(
        ExpectedTensor{prefix + "mlp.gate_proj.weight", gate_up.value()});
    expected.push_back(
        ExpectedTensor{prefix + "mlp.up_proj.weight", gate_up.value()});
    expected.push_back(
        ExpectedTensor{prefix + "mlp.down_proj.weight", down.value()});
  }
  expected.push_back(ExpectedTensor{"model.norm.weight", norm.value()});
  return Result<std::vector<ExpectedTensor>>::success(std::move(expected));
}

Result<ConstTensorView> require_tensor(const SafeTensorFile& file,
                                       const ExpectedTensor& expected,
                                       const std::uint32_t detail) {
  const auto* const record = file.record(expected.name);
  if (record == nullptr) {
    return Result<ConstTensorView>::failure(
        Status::failure(ErrorCode::missing_tensor, detail));
  }
  if (record->shape != expected.shape || record->dtype != DType::bf16) {
    return Result<ConstTensorView>::failure(
        Status::failure(ErrorCode::invalid_tensor, detail));
  }
  return file.tensor(expected.name);
}

} // namespace

Result<LoadedWeights>
LoadedWeights::open_and_bind(const std::filesystem::path& path,
                             const ModelSpec& spec) {
  auto mapping = MappedFile::open_read_only(path);
  if (!mapping) {
    return Result<LoadedWeights>::failure(mapping.status());
  }
  auto file = SafeTensorFile::parse(std::move(mapping).value());
  if (!file) {
    return Result<LoadedWeights>::failure(file.status());
  }
  return bind(std::move(file).value(), spec);
}

Result<LoadedWeights> LoadedWeights::bind(SafeTensorFile file,
                                          const ModelSpec& spec) {
  if (spec.architecture != Architecture::smollm2_llama || spec.qkv_bias ||
      !spec.tied_embeddings || !validate(spec).ok()) {
    return Result<LoadedWeights>::failure(
        Status::failure(ErrorCode::invalid_model));
  }

  auto expected_result = smollm2_expected_tensors(spec);
  if (!expected_result) {
    return Result<LoadedWeights>::failure(expected_result.status());
  }
  const auto& expected = expected_result.value();
  std::unordered_set<std::string_view> allowed;
  allowed.reserve(expected.size() + 1);
  for (const auto& tensor : expected) {
    allowed.insert(tensor.name);
  }
  allowed.insert("lm_head.weight");

  for (std::size_t index = 0; index < file.records().size(); ++index) {
    if (!allowed.contains(file.records()[index].name)) {
      return Result<LoadedWeights>::failure(Status::failure(
          ErrorCode::unexpected_tensor, static_cast<std::uint32_t>(index + 1)));
    }
  }

  std::vector<ConstTensorView> views;
  views.reserve(expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    auto view = require_tensor(file, expected[index],
                               static_cast<std::uint32_t>(index + 1));
    if (!view) {
      return Result<LoadedWeights>::failure(view.status());
    }
    views.push_back(view.value());
  }

  const ConstTensorView embedding = views.front();
  std::vector<LayerWeights> layers;
  layers.reserve(spec.layers);
  std::size_t cursor = 1;
  for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
    static_cast<void>(layer);
    const ConstTensorView input_norm = views[cursor++];
    const ConstTensorView query = views[cursor++];
    const ConstTensorView key = views[cursor++];
    const ConstTensorView value = views[cursor++];
    const ConstTensorView output = views[cursor++];
    const ConstTensorView post_norm = views[cursor++];
    const ConstTensorView gate = views[cursor++];
    const ConstTensorView up = views[cursor++];
    const ConstTensorView down = views[cursor++];
    layers.push_back(LayerWeights{
        input_norm,
        AttentionWeights{query, key, value, output},
        post_norm,
        MlpWeights{gate, up, down},
    });
  }
  const ConstTensorView final_norm = views[cursor];

  ConstTensorView output_head = embedding;
  bool aliases_embedding = true;
  if (const auto* const output_record = file.record("lm_head.weight");
      output_record != nullptr) {
    const auto expected_output =
        shape_of({spec.vocabulary_size, spec.hidden_size});
    if (!expected_output || output_record->shape != expected_output.value() ||
        output_record->dtype != DType::bf16) {
      return Result<LoadedWeights>::failure(
          Status::failure(ErrorCode::invalid_tensor,
                          static_cast<std::uint32_t>(expected.size() + 1)));
    }
    output_head = file.tensor("lm_head.weight").value();
    aliases_embedding = false;
  }

  return Result<LoadedWeights>::success(LoadedWeights{
      std::move(file),
      embedding,
      final_norm,
      output_head,
      aliases_embedding,
      std::move(layers),
  });
}

} // namespace marketforge
