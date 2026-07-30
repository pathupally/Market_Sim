#include "marketforge/cuda/decoder_layer.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

#include "marketforge/cuda/attention.hpp"
#include "marketforge/cuda/elementwise.hpp"
#include "marketforge/cuda/kv_cache.hpp"
#include "marketforge/cuda/linear.hpp"
#include "marketforge/cuda/rms_norm.hpp"
#include "marketforge/cuda/rope.hpp"
#include "marketforge/cuda/swiglu.hpp"

namespace marketforge::cuda {

Status smollm2_decoder_layer_f16(
    const CudaDecoderLayerWeightsView& weights,
    const CudaDecoderLayerSpec& spec, DeviceBuffer& hidden,
    const CudaKvCacheView cache, const DeviceBuffer& positions,
    const CudaDecoderLayerWorkspaceView workspace,
    const std::uint64_t batch, const std::uint64_t tokens,
    const std::uint64_t maximum_context, CublasHandle& cublas,
    const StreamHandle stream) noexcept {
  if (batch == 0 || tokens == 0 || spec.hidden_size == 0 ||
      spec.intermediate_size == 0 || spec.query_heads == 0 ||
      spec.key_value_heads == 0 ||
      spec.query_heads % spec.key_value_heads != 0 ||
      spec.head_dim == 0 ||
      spec.query_heads >
          std::numeric_limits<std::uint64_t>::max() / spec.head_dim ||
      spec.query_heads * spec.head_dim != spec.hidden_size ||
      !std::isfinite(spec.rms_norm_epsilon) ||
      spec.rms_norm_epsilon <= 0.0F || !std::isfinite(spec.rope_theta) ||
      spec.rope_theta <= 0.0F || !cublas.handle().valid() ||
      !stream.valid()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (batch > std::numeric_limits<std::uint64_t>::max() / tokens) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto rows = batch * tokens;
  if (rows > std::numeric_limits<std::uint64_t>::max() / spec.hidden_size ||
      rows >
          std::numeric_limits<std::uint64_t>::max() /
              spec.intermediate_size ||
      spec.key_value_heads >
          std::numeric_limits<std::uint64_t>::max() / spec.head_dim) {
    return Status::failure(ErrorCode::arithmetic_overflow);
  }
  const auto hidden_elements = rows * spec.hidden_size;
  const auto intermediate_elements = rows * spec.intermediate_size;
  const auto key_value_width = spec.key_value_heads * spec.head_dim;

  auto status = rms_norm_f16(
      hidden, weights.input_norm, workspace.normalized, rows,
      spec.hidden_size, spec.rms_norm_epsilon, stream);
  if (status.ok()) {
    status = linear_f16(
        workspace.normalized, weights.attention.query, workspace.query, rows,
        spec.hidden_size, spec.hidden_size, cublas, stream);
  }
  if (status.ok()) {
    status = linear_f16(
        workspace.normalized, weights.attention.key, workspace.key, rows,
        spec.hidden_size, key_value_width, cublas, stream);
  }
  if (status.ok()) {
    status = linear_f16(
        workspace.normalized, weights.attention.value, workspace.value, rows,
        spec.hidden_size, key_value_width, cublas, stream);
  }
  if (status.ok()) {
    status = apply_rope_f16(
        workspace.query, workspace.key, positions, batch, tokens,
        spec.query_heads, spec.key_value_heads, spec.head_dim,
        spec.rope_theta, stream);
  }
  if (status.ok()) {
    status = append_kv_f16(
        workspace.key, workspace.value, positions, cache.key, cache.value,
        batch, tokens, maximum_context, spec.key_value_heads, spec.head_dim,
        stream);
  }
  if (status.ok()) {
    status = attention_f16(
        workspace.query, cache.key, cache.value, positions,
        workspace.attention, batch, tokens, maximum_context,
        spec.query_heads, spec.key_value_heads, spec.head_dim, stream);
  }
  if (status.ok()) {
    status = linear_f16(
        workspace.attention, weights.attention.output, workspace.normalized,
        rows, spec.hidden_size, spec.hidden_size, cublas, stream);
  }
  if (status.ok()) {
    status = add_f16(
        hidden, workspace.normalized, hidden, hidden_elements, stream);
  }
  if (status.ok()) {
    status = rms_norm_f16(
        hidden, weights.post_attention_norm, workspace.normalized, rows,
        spec.hidden_size, spec.rms_norm_epsilon, stream);
  }
  if (status.ok()) {
    status = linear_f16(
        workspace.normalized, weights.mlp.gate, workspace.gate, rows,
        spec.hidden_size, spec.intermediate_size, cublas, stream);
  }
  if (status.ok()) {
    status = linear_f16(
        workspace.normalized, weights.mlp.up, workspace.up, rows,
        spec.hidden_size, spec.intermediate_size, cublas, stream);
  }
  if (status.ok()) {
    status = swiglu_f16(
        workspace.gate, workspace.up, workspace.gate,
        intermediate_elements, stream);
  }
  if (status.ok()) {
    status = linear_f16(
        workspace.gate, weights.mlp.down, workspace.attention, rows,
        spec.intermediate_size, spec.hidden_size, cublas, stream);
  }
  if (status.ok()) {
    status =
        add_f16(hidden, workspace.attention, hidden, hidden_elements, stream);
  }
  return status;
}

} // namespace marketforge::cuda
