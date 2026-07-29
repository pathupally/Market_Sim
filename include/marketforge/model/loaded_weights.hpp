#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

#include "marketforge/model/model_spec.hpp"
#include "marketforge/model/safetensors.hpp"

namespace marketforge {

struct AttentionWeights {
  ConstTensorView query;
  ConstTensorView key;
  ConstTensorView value;
  ConstTensorView output;
};

struct MlpWeights {
  ConstTensorView gate;
  ConstTensorView up;
  ConstTensorView down;
};

struct LayerWeights {
  ConstTensorView input_norm;
  AttentionWeights attention;
  ConstTensorView post_attention_norm;
  MlpWeights mlp;
};

class LoadedWeights {
public:
  [[nodiscard]] static Result<LoadedWeights>
  open_and_bind(const std::filesystem::path& path, const ModelSpec& spec);

  [[nodiscard]] static Result<LoadedWeights> bind(SafeTensorFile file,
                                                  const ModelSpec& spec);

  LoadedWeights(LoadedWeights&&) noexcept = default;
  LoadedWeights& operator=(LoadedWeights&&) noexcept = default;
  LoadedWeights(const LoadedWeights&) = delete;
  LoadedWeights& operator=(const LoadedWeights&) = delete;

  [[nodiscard]] ConstTensorView embedding() const noexcept {
    return embedding_;
  }
  [[nodiscard]] ConstTensorView final_norm() const noexcept {
    return final_norm_;
  }
  [[nodiscard]] ConstTensorView output_head() const noexcept {
    return output_head_;
  }
  [[nodiscard]] bool output_head_aliases_embedding() const noexcept {
    return output_head_aliases_embedding_;
  }
  [[nodiscard]] std::span<const LayerWeights> layers() const noexcept {
    return layers_;
  }

private:
  LoadedWeights(SafeTensorFile file, ConstTensorView embedding,
                ConstTensorView final_norm, ConstTensorView output_head,
                bool output_head_aliases_embedding,
                std::vector<LayerWeights> layers) noexcept
      : file_(std::move(file)), embedding_(embedding), final_norm_(final_norm),
        output_head_(output_head),
        output_head_aliases_embedding_(output_head_aliases_embedding),
        layers_(std::move(layers)) {}

  SafeTensorFile file_;
  ConstTensorView embedding_;
  ConstTensorView final_norm_;
  ConstTensorView output_head_;
  bool output_head_aliases_embedding_{false};
  std::vector<LayerWeights> layers_;
};

} // namespace marketforge
