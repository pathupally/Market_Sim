#include "marketforge/cuda/smollm2.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cuda_internal.hpp"
#include "marketforge/core/checked_math.hpp"
#include "marketforge/core/dtype.hpp"
#include "marketforge/cuda/decoder_layer.hpp"
#include "marketforge/cuda/embedding.hpp"
#include "marketforge/cuda/greedy.hpp"
#include "marketforge/cuda/linear.hpp"
#include "marketforge/cuda/rms_norm.hpp"
#include "marketforge/model/loaded_weights.hpp"

namespace marketforge::cuda {
namespace {

float decode_bf16(const std::byte* const bytes) noexcept {
  const auto low =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0]));
  const auto high =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1]));
  const std::uint16_t bf16 =
      static_cast<std::uint16_t>(low | static_cast<std::uint16_t>(high << 8U));
  return std::bit_cast<float>(static_cast<std::uint32_t>(bf16) << 16U);
}

Result<DeviceBuffer> upload_bf16(const ConstTensorView tensor) noexcept {
  if (tensor.dtype != DType::bf16 || tensor.memory != MemoryKind::host ||
      tensor.data == nullptr) {
    return Result<DeviceBuffer>::failure(
        Status::failure(ErrorCode::invalid_tensor));
  }
  const auto elements = checked_numel(tensor.shape);
  if (!elements ||
      elements.value() >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half)) {
    return Result<DeviceBuffer>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  const auto bytes = elements.value() * sizeof(__half);
  if (elements.value() > std::numeric_limits<std::size_t>::max()) {
    return Result<DeviceBuffer>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  std::vector<__half> converted(
      static_cast<std::size_t>(elements.value()));
  const auto* source = static_cast<const std::byte*>(tensor.data);
  for (std::size_t index = 0; index < converted.size(); ++index) {
    converted[index] = __float2half_rn(
        decode_bf16(source + index * sizeof(std::uint16_t)));
  }
  auto allocation = DeviceBuffer::allocate(bytes);
  if (!allocation) {
    return allocation;
  }
  DeviceBuffer result = std::move(allocation).value();
  const auto copy_status = detail::runtime_status(cudaMemcpy(
      detail::native_address(result.address()), converted.data(),
      static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
  if (!copy_status.ok()) {
    return Result<DeviceBuffer>::failure(copy_status);
  }
  return Result<DeviceBuffer>::success(std::move(result));
}

Result<DeviceBuffer> allocate_f16(const std::uint64_t elements) noexcept {
  if (elements == 0 ||
      elements >
          std::numeric_limits<std::uint64_t>::max() / sizeof(__half)) {
    return Result<DeviceBuffer>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return DeviceBuffer::allocate(elements * sizeof(__half));
}

Status zero(DeviceBuffer& buffer) noexcept {
  return detail::runtime_status(cudaMemset(
      detail::native_address(buffer.address()), 0,
      static_cast<std::size_t>(buffer.size_bytes())));
}

} // namespace

Result<CudaSmolLm2::ExecutionBuffers>
CudaSmolLm2::allocate_execution(const std::uint64_t rows,
                                const ModelSpec& spec) noexcept {
  const auto hidden_elements = checked_multiply(rows, spec.hidden_size);
  const auto key_value_width =
      checked_multiply(spec.kv_heads, spec.head_dim);
  const auto key_value_elements =
      key_value_width
          ? checked_multiply(rows, key_value_width.value())
          : Result<std::uint64_t>::failure(key_value_width.status());
  const auto intermediate_elements =
      checked_multiply(rows, spec.intermediate_size);
  if (!hidden_elements || !key_value_elements || !intermediate_elements ||
      rows > std::numeric_limits<std::uint64_t>::max() /
                 sizeof(std::uint32_t)) {
    return Result<ExecutionBuffers>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  auto token_ids =
      DeviceBuffer::allocate(rows * sizeof(std::uint32_t));
  auto positions =
      DeviceBuffer::allocate(rows * sizeof(std::uint32_t));
  auto hidden = allocate_f16(hidden_elements.value());
  auto normalized = allocate_f16(hidden_elements.value());
  auto query = allocate_f16(hidden_elements.value());
  auto key = allocate_f16(key_value_elements.value());
  auto value = allocate_f16(key_value_elements.value());
  auto attention = allocate_f16(hidden_elements.value());
  auto gate = allocate_f16(intermediate_elements.value());
  auto up = allocate_f16(intermediate_elements.value());
  if (!token_ids || !positions || !hidden || !normalized || !query || !key ||
      !value || !attention || !gate || !up) {
    return Result<ExecutionBuffers>::failure(
        Status::failure(ErrorCode::allocation_failed));
  }
  return Result<ExecutionBuffers>::success(ExecutionBuffers{
      std::move(token_ids).value(),
      std::move(positions).value(),
      std::move(hidden).value(),
      std::move(normalized).value(),
      std::move(query).value(),
      std::move(key).value(),
      std::move(value).value(),
      std::move(attention).value(),
      std::move(gate).value(),
      std::move(up).value(),
      rows,
  });
}

Result<CudaSmolLm2>
CudaSmolLm2::load(const std::filesystem::path& checkpoint,
                  const std::uint32_t maximum_context,
                  const std::uint32_t maximum_prefill_tokens,
                  const std::uint32_t maximum_restricted_tokens) {
  const ModelSpec spec = smollm2_135m_profile().spec;
  if (maximum_context == 0 || maximum_context > spec.max_positions ||
      maximum_prefill_tokens == 0 ||
      maximum_prefill_tokens > maximum_context ||
      maximum_restricted_tokens == 0) {
    return Result<CudaSmolLm2>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  auto source = LoadedWeights::open_and_bind(checkpoint, spec);
  if (!source || !source.value().output_head_aliases_embedding() ||
      source.value().layers().size() != spec.layers) {
    return Result<CudaSmolLm2>::failure(
        source ? Status::failure(ErrorCode::invalid_model) : source.status());
  }
  auto stream = CudaStream::create();
  auto cublas = CublasHandle::create();
  auto embedding = upload_bf16(source.value().embedding());
  auto final_norm = upload_bf16(source.value().final_norm());
  if (!stream || !cublas || !embedding || !final_norm) {
    return Result<CudaSmolLm2>::failure(
        Status::failure(ErrorCode::allocation_failed));
  }

  CudaSmolLm2 model;
  model.spec_ = spec;
  model.stream_ = std::move(stream).value();
  model.cublas_ = std::move(cublas).value();
  model.embedding_ = std::move(embedding).value();
  model.final_norm_ = std::move(final_norm).value();
  model.maximum_context_ = maximum_context;
  model.maximum_prefill_tokens_ = maximum_prefill_tokens;
  model.maximum_restricted_tokens_ = maximum_restricted_tokens;
  model.host_positions_.resize(maximum_prefill_tokens);
  model.layers_.reserve(spec.layers);

  const auto cache_elements =
      checked_multiply(
          maximum_context,
          static_cast<std::uint64_t>(spec.kv_heads) * spec.head_dim);
  if (!cache_elements) {
    return Result<CudaSmolLm2>::failure(cache_elements.status());
  }
  for (const auto& layer : source.value().layers()) {
    auto input_norm = upload_bf16(layer.input_norm);
    auto query = upload_bf16(layer.attention.query);
    auto key = upload_bf16(layer.attention.key);
    auto value = upload_bf16(layer.attention.value);
    auto output = upload_bf16(layer.attention.output);
    auto post_norm = upload_bf16(layer.post_attention_norm);
    auto gate = upload_bf16(layer.mlp.gate);
    auto up = upload_bf16(layer.mlp.up);
    auto down = upload_bf16(layer.mlp.down);
    auto key_cache = allocate_f16(cache_elements.value());
    auto value_cache = allocate_f16(cache_elements.value());
    if (!input_norm || !query || !key || !value || !output || !post_norm ||
        !gate || !up || !down || !key_cache || !value_cache) {
      return Result<CudaSmolLm2>::failure(
          Status::failure(ErrorCode::allocation_failed));
    }
    LayerStorage storage{
        std::move(input_norm).value(),
        std::move(query).value(),
        std::move(key).value(),
        std::move(value).value(),
        std::move(output).value(),
        std::move(post_norm).value(),
        std::move(gate).value(),
        std::move(up).value(),
        std::move(down).value(),
        std::move(key_cache).value(),
        std::move(value_cache).value(),
    };
    const auto key_zero = zero(storage.key_cache);
    const auto value_zero = zero(storage.value_cache);
    if (!key_zero.ok() || !value_zero.ok()) {
      return Result<CudaSmolLm2>::failure(
          !key_zero.ok() ? key_zero : value_zero);
    }
    model.layers_.push_back(std::move(storage));
  }

  auto prefill = allocate_execution(maximum_prefill_tokens, spec);
  auto decode = allocate_execution(1, spec);
  auto last_hidden = allocate_f16(spec.hidden_size);
  auto last_normalized = allocate_f16(spec.hidden_size);
  auto logits = allocate_f16(spec.vocabulary_size);
  auto restricted_token_ids = DeviceBuffer::allocate(
      static_cast<std::uint64_t>(maximum_restricted_tokens) *
      sizeof(std::uint32_t));
  auto restricted_token_count =
      DeviceBuffer::allocate(sizeof(std::uint32_t));
  auto selected_token =
      DeviceBuffer::allocate(sizeof(std::uint32_t));
  if (!prefill || !decode || !last_hidden || !last_normalized || !logits ||
      !restricted_token_ids || !restricted_token_count || !selected_token) {
    return Result<CudaSmolLm2>::failure(
        Status::failure(ErrorCode::allocation_failed));
  }
  model.prefill_ = std::move(prefill).value();
  model.decode_ = std::move(decode).value();
  model.last_hidden_ = std::move(last_hidden).value();
  model.last_normalized_ = std::move(last_normalized).value();
  model.logits_ = std::move(logits).value();
  model.restricted_token_ids_ = std::move(restricted_token_ids).value();
  model.restricted_token_count_ =
      std::move(restricted_token_count).value();
  model.selected_token_ = std::move(selected_token).value();

  const auto parameter_count = estimate_parameter_count(spec);
  const auto weight_bytes =
      parameter_count
          ? checked_multiply(parameter_count.value(), sizeof(__half))
          : Result<std::uint64_t>::failure(parameter_count.status());
  const auto per_layer_cache =
      checked_multiply(cache_elements.value(), 2 * sizeof(__half));
  const auto kv_bytes =
      per_layer_cache
          ? checked_multiply(per_layer_cache.value(), spec.layers)
          : Result<std::uint64_t>::failure(per_layer_cache.status());
  if (!weight_bytes || !kv_bytes) {
    return Result<CudaSmolLm2>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  std::uint64_t execution_bytes =
      model.prefill_.token_ids.size_bytes() +
      model.prefill_.positions.size_bytes() +
      model.prefill_.hidden.size_bytes() +
      model.prefill_.normalized.size_bytes() +
      model.prefill_.query.size_bytes() +
      model.prefill_.key.size_bytes() +
      model.prefill_.value.size_bytes() +
      model.prefill_.attention.size_bytes() +
      model.prefill_.gate.size_bytes() + model.prefill_.up.size_bytes() +
      model.decode_.token_ids.size_bytes() +
      model.decode_.positions.size_bytes() +
      model.decode_.hidden.size_bytes() +
      model.decode_.normalized.size_bytes() +
      model.decode_.query.size_bytes() + model.decode_.key.size_bytes() +
      model.decode_.value.size_bytes() +
      model.decode_.attention.size_bytes() +
      model.decode_.gate.size_bytes() + model.decode_.up.size_bytes() +
      model.last_hidden_.size_bytes() +
      model.last_normalized_.size_bytes() + model.logits_.size_bytes() +
      model.restricted_token_ids_.size_bytes() +
      model.restricted_token_count_.size_bytes() +
      model.selected_token_.size_bytes();
  const auto weights_and_kv =
      checked_add(weight_bytes.value(), kv_bytes.value());
  const auto total =
      weights_and_kv
          ? checked_add(weights_and_kv.value(), execution_bytes)
          : Result<std::uint64_t>::failure(weights_and_kv.status());
  if (!total) {
    return Result<CudaSmolLm2>::failure(total.status());
  }
  model.memory_ = {
      weight_bytes.value(),
      kv_bytes.value(),
      execution_bytes,
      total.value(),
  };
  return Result<CudaSmolLm2>::success(std::move(model));
}

Status CudaSmolLm2::reset() noexcept {
  for (auto& layer : layers_) {
    const auto key_status = zero(layer.key_cache);
    const auto value_status = zero(layer.value_cache);
    if (!key_status.ok() || !value_status.ok()) {
      return !key_status.ok() ? key_status : value_status;
    }
  }
  context_length_ = 0;
  return Status::success();
}

Result<std::uint32_t>
CudaSmolLm2::prefill(
    const std::span<const std::uint32_t> token_ids) noexcept {
  if (context_length_ != 0 ||
      token_ids.size() != maximum_prefill_tokens_) {
    return Result<std::uint32_t>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return forward(token_ids, prefill_, {});
}

Result<std::uint32_t> CudaSmolLm2::prefill_restricted(
    const std::span<const std::uint32_t> token_ids,
    const std::span<const std::uint32_t> allowed_token_ids) noexcept {
  if (context_length_ != 0 ||
      token_ids.size() != maximum_prefill_tokens_ ||
      allowed_token_ids.empty()) {
    return Result<std::uint32_t>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return forward(token_ids, prefill_, allowed_token_ids);
}

Result<std::uint32_t>
CudaSmolLm2::decode(const std::uint32_t token_id) noexcept {
  if (context_length_ == 0) {
    return Result<std::uint32_t>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return forward(std::span<const std::uint32_t>(&token_id, 1), decode_, {});
}

Result<std::uint32_t> CudaSmolLm2::decode_restricted(
    const std::uint32_t token_id,
    const std::span<const std::uint32_t> allowed_token_ids) noexcept {
  if (context_length_ == 0 || allowed_token_ids.empty()) {
    return Result<std::uint32_t>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return forward(std::span<const std::uint32_t>(&token_id, 1), decode_,
                 allowed_token_ids);
}

Result<std::uint32_t> CudaSmolLm2::forward(
    const std::span<const std::uint32_t> token_ids,
    ExecutionBuffers& execution,
    const std::span<const std::uint32_t> allowed_token_ids) noexcept {
  if (token_ids.size() != execution.rows ||
      execution.rows > maximum_context_ - context_length_ ||
      allowed_token_ids.size() > maximum_restricted_tokens_) {
    return Result<std::uint32_t>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  for (std::size_t index = 0; index < token_ids.size(); ++index) {
    if (token_ids[index] >= spec_.vocabulary_size) {
      return Result<std::uint32_t>::failure(
          Status::failure(ErrorCode::invalid_argument, token_ids[index]));
    }
    host_positions_[index] =
        context_length_ + static_cast<std::uint32_t>(index);
  }
  for (const auto token_id : allowed_token_ids) {
    if (token_id >= spec_.vocabulary_size) {
      return Result<std::uint32_t>::failure(
          Status::failure(ErrorCode::invalid_argument, token_id));
    }
  }
  if (!allowed_token_ids.empty()) {
    const auto count =
        static_cast<std::uint32_t>(allowed_token_ids.size());
    auto restricted_status = restricted_token_ids_.copy_from_host_async(
        allowed_token_ids.data(),
        allowed_token_ids.size_bytes(), 0, stream_.handle());
    if (restricted_status.ok()) {
      restricted_status = restricted_token_count_.copy_from_host_async(
          &count, sizeof(count), 0, stream_.handle());
    }
    if (!restricted_status.ok()) {
      return Result<std::uint32_t>::failure(restricted_status);
    }
  }
  const auto token_bytes =
      execution.rows * sizeof(std::uint32_t);
  auto status = execution.token_ids.copy_from_host_async(
      token_ids.data(), token_bytes, 0, stream_.handle());
  if (status.ok()) {
    status = execution.positions.copy_from_host_async(
        host_positions_.data(), token_bytes, 0, stream_.handle());
  }
  if (status.ok()) {
    status = embedding_lookup_f16(
        embedding_, execution.token_ids, execution.hidden, execution.rows,
        spec_.vocabulary_size, spec_.hidden_size, stream_.handle());
  }
  const CudaDecoderLayerSpec layer_spec{
      spec_.hidden_size,
      spec_.intermediate_size,
      spec_.query_heads,
      spec_.kv_heads,
      spec_.head_dim,
      spec_.rms_norm_epsilon,
      spec_.rope_theta,
  };
  for (std::size_t index = 0; status.ok() && index < layers_.size(); ++index) {
    auto& layer = layers_[index];
    const CudaDecoderLayerWeightsView weights{
        layer.input_norm,
        {
            layer.query,
            layer.key,
            layer.value,
            layer.output,
        },
        layer.post_attention_norm,
        {
            layer.gate,
            layer.up,
            layer.down,
        },
    };
    const CudaDecoderLayerWorkspaceView workspace{
        execution.normalized,
        execution.query,
        execution.key,
        execution.value,
        execution.attention,
        execution.gate,
        execution.up,
    };
    status = smollm2_decoder_layer_f16(
        weights, layer_spec, execution.hidden,
        CudaKvCacheView{layer.key_cache, layer.value_cache},
        execution.positions, workspace, 1, execution.rows, maximum_context_,
        cublas_, stream_.handle());
  }
  const auto hidden_bytes =
      static_cast<std::size_t>(spec_.hidden_size * sizeof(__half));
  const auto last_offset =
      static_cast<std::size_t>(
          (execution.rows - 1) * spec_.hidden_size * sizeof(__half));
  if (status.ok()) {
    auto* source = static_cast<const std::byte*>(
        detail::native_address(execution.hidden.address()));
    status = detail::runtime_status(cudaMemcpyAsync(
        detail::native_address(last_hidden_.address()), source + last_offset,
        hidden_bytes, cudaMemcpyDeviceToDevice,
        detail::native_stream(stream_.handle())));
  }
  if (status.ok()) {
    status = rms_norm_f16(
        last_hidden_, final_norm_, last_normalized_, 1, spec_.hidden_size,
        spec_.rms_norm_epsilon, stream_.handle());
  }
  if (status.ok() && allowed_token_ids.empty()) {
    status = linear_f16(
        last_normalized_, embedding_, logits_, 1, spec_.hidden_size,
        spec_.vocabulary_size, cublas_, stream_.handle());
  }
  if (status.ok() && allowed_token_ids.empty()) {
    status = greedy_select_f16(
        logits_, selected_token_, 1, spec_.vocabulary_size,
        stream_.handle());
  }
  if (status.ok() && !allowed_token_ids.empty()) {
    status = restricted_output_head_f16(
        last_normalized_, embedding_, restricted_token_ids_,
        restricted_token_count_, selected_token_, 1, spec_.hidden_size,
        spec_.vocabulary_size, maximum_restricted_tokens_, stream_.handle());
  }
  std::uint32_t selected = 0;
  if (status.ok()) {
    status = selected_token_.copy_to_host_async(
        &selected, sizeof(selected), 0, stream_.handle());
  }
  if (status.ok()) {
    status = stream_.synchronize();
  }
  if (!status.ok()) {
    return Result<std::uint32_t>::failure(status);
  }
  context_length_ += static_cast<std::uint32_t>(execution.rows);
  return Result<std::uint32_t>::success(selected);
}

} // namespace marketforge::cuda
