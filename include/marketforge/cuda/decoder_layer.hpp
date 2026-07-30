#pragma once

#include <cstdint>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"

namespace marketforge::cuda {

struct CudaDecoderLayerSpec {
  std::uint64_t hidden_size{0};
  std::uint64_t intermediate_size{0};
  std::uint64_t query_heads{0};
  std::uint64_t key_value_heads{0};
  std::uint64_t head_dim{0};
  float rms_norm_epsilon{0.0F};
  float rope_theta{0.0F};
};

struct CudaAttentionWeightsView {
  const DeviceBuffer& query;
  const DeviceBuffer& key;
  const DeviceBuffer& value;
  const DeviceBuffer& output;
};

struct CudaMlpWeightsView {
  const DeviceBuffer& gate;
  const DeviceBuffer& up;
  const DeviceBuffer& down;
};

struct CudaDecoderLayerWeightsView {
  const DeviceBuffer& input_norm;
  CudaAttentionWeightsView attention;
  const DeviceBuffer& post_attention_norm;
  CudaMlpWeightsView mlp;
};

struct CudaDecoderLayerWorkspaceView {
  DeviceBuffer& normalized;
  DeviceBuffer& query;
  DeviceBuffer& key;
  DeviceBuffer& value;
  DeviceBuffer& attention;
  DeviceBuffer& gate;
  DeviceBuffer& up;
};

struct CudaKvCacheView {
  DeviceBuffer& key;
  DeviceBuffer& value;
};

// Runs one bias-free Llama decoder layer in place using fixed caller-owned
// workspace and contiguous KV storage.
[[nodiscard]] Status smollm2_decoder_layer_f16(
    const CudaDecoderLayerWeightsView& weights,
    const CudaDecoderLayerSpec& spec, DeviceBuffer& hidden,
    CudaKvCacheView cache, const DeviceBuffer& positions,
    CudaDecoderLayerWorkspaceView workspace, std::uint64_t batch,
    std::uint64_t tokens, std::uint64_t maximum_context,
    CublasHandle& cublas, StreamHandle stream) noexcept;

} // namespace marketforge::cuda
