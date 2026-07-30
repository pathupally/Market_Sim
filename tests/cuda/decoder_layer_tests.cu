#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <cuda_fp16.h>

#include "marketforge/core/dtype.hpp"
#include "marketforge/core/shape.hpp"
#include "marketforge/core/tensor_view.hpp"
#include "marketforge/cpu/operators.hpp"
#include "marketforge/cuda/attention.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/decoder_layer.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/embedding.hpp"
#include "test_support.hpp"

namespace {

using marketforge::ConstTensorView;
using marketforge::CpuKvView;
using marketforge::CpuWorkspace;
using marketforge::DType;
using marketforge::LayerWeights;
using marketforge::MemoryKind;
using marketforge::TensorView;
using marketforge::cuda::CublasHandle;
using marketforge::cuda::CudaAttentionWeightsView;
using marketforge::cuda::CudaDecoderLayerSpec;
using marketforge::cuda::CudaDecoderLayerWeightsView;
using marketforge::cuda::CudaDecoderLayerWorkspaceView;
using marketforge::cuda::CudaKvCacheView;
using marketforge::cuda::CudaMlpWeightsView;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

std::vector<__half> make_half_values(const std::size_t elements,
                                     const std::uint32_t modulus,
                                     const float scale) {
  std::vector<__half> values(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const auto centered =
        static_cast<std::int32_t>(index % modulus) -
        static_cast<std::int32_t>(modulus / 2);
    values[index] =
        __float2half_rn(static_cast<float>(centered) * scale);
  }
  return values;
}

std::vector<float> as_float(const std::vector<__half>& values) {
  std::vector<float> result(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    result[index] = __half2float(values[index]);
  }
  return result;
}

DeviceBuffer copy_to_device(const std::vector<__half>& values,
                            const CudaStream& stream) {
  auto result = DeviceBuffer::allocate(values.size() * sizeof(__half));
  MF_CHECK(result);
  DeviceBuffer buffer = std::move(result).value();
  MF_CHECK(buffer
               .copy_from_host_async(
                   values.data(), values.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  return buffer;
}

ConstTensorView const_view(
    const std::vector<float>& values,
    const std::span<const std::uint64_t> extents) {
  const auto shape = marketforge::make_shape(extents);
  MF_CHECK(shape);
  return {
      values.data(),
      shape.value(),
      DType::f32,
      MemoryKind::host,
  };
}

TensorView view(std::vector<float>& values,
                const std::span<const std::uint64_t> extents) {
  const auto shape = marketforge::make_shape(extents);
  MF_CHECK(shape);
  return {
      values.data(),
      shape.value(),
      DType::f32,
      MemoryKind::host,
  };
}

MF_TEST(cuda_embedding_lookup_f16_gathers_packed_rows) {
  constexpr std::uint64_t vocabulary = 4;
  constexpr std::uint64_t hidden = 3;
  const auto embedding =
      make_half_values(vocabulary * hidden, 13, 0.125F);
  const std::vector<std::uint32_t> tokens{2, 0};
  std::vector<__half> observed(tokens.size() * hidden);
  auto stream_result = CudaStream::create();
  auto token_result =
      DeviceBuffer::allocate(tokens.size() * sizeof(std::uint32_t));
  auto output_result =
      DeviceBuffer::allocate(observed.size() * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(token_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer embedding_device = copy_to_device(embedding, stream);
  DeviceBuffer token_device = std::move(token_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  MF_CHECK(token_device
               .copy_from_host_async(
                   tokens.data(), tokens.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::embedding_lookup_f16(
               embedding_device, token_device, output_device, tokens.size(),
               vocabulary, hidden, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(), observed.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t token = 0; token < tokens.size(); ++token) {
    for (std::size_t column = 0; column < hidden; ++column) {
      const auto expected =
          embedding[static_cast<std::size_t>(tokens[token]) * hidden + column];
      MF_CHECK_EQ(__half2float(observed[token * hidden + column]),
                  __half2float(expected));
    }
  }
}

MF_TEST(cuda_decoder_layer_f16_matches_rounded_fp32_oracle) {
  constexpr std::uint64_t batch = 1;
  constexpr std::uint64_t tokens = 2;
  constexpr std::uint64_t hidden_size = 4;
  constexpr std::uint64_t intermediate_size = 6;
  constexpr std::uint64_t query_heads = 2;
  constexpr std::uint64_t key_value_heads = 1;
  constexpr std::uint64_t head_dim = 2;
  constexpr std::uint64_t maximum_context = 4;
  constexpr std::uint64_t rows = batch * tokens;
  constexpr std::uint64_t key_value_width = key_value_heads * head_dim;
  const std::vector<std::uint32_t> positions{0, 1};

  auto hidden_half =
      make_half_values(rows * hidden_size, 11, 0.0625F);
  auto input_norm_half =
      make_half_values(hidden_size, 7, 0.03125F);
  auto query_weight_half =
      make_half_values(hidden_size * hidden_size, 13, 0.03125F);
  auto key_weight_half =
      make_half_values(key_value_width * hidden_size, 11, 0.03125F);
  auto value_weight_half =
      make_half_values(key_value_width * hidden_size, 9, 0.03125F);
  auto output_weight_half =
      make_half_values(hidden_size * hidden_size, 15, 0.03125F);
  auto post_norm_half =
      make_half_values(hidden_size, 9, 0.03125F);
  auto gate_weight_half =
      make_half_values(intermediate_size * hidden_size, 17, 0.015625F);
  auto up_weight_half =
      make_half_values(intermediate_size * hidden_size, 19, 0.015625F);
  auto down_weight_half =
      make_half_values(hidden_size * intermediate_size, 21, 0.015625F);
  for (auto* norm : {&input_norm_half, &post_norm_half}) {
    for (std::size_t index = 0; index < norm->size(); ++index) {
      (*norm)[index] = __float2half_rn(
          0.75F + static_cast<float>(index) * 0.0625F);
    }
  }

  auto expected_hidden = as_float(hidden_half);
  auto input_norm = as_float(input_norm_half);
  auto query_weight = as_float(query_weight_half);
  auto key_weight = as_float(key_weight_half);
  auto value_weight = as_float(value_weight_half);
  auto output_weight = as_float(output_weight_half);
  auto post_norm = as_float(post_norm_half);
  auto gate_weight = as_float(gate_weight_half);
  auto up_weight = as_float(up_weight_half);
  auto down_weight = as_float(down_weight_half);
  std::vector<float> expected_key_cache(
      batch * maximum_context * key_value_width, 0.0F);
  std::vector<float> expected_value_cache(
      batch * maximum_context * key_value_width, 0.0F);
  const std::array<std::uint64_t, 1> norm_shape{hidden_size};
  const std::array<std::uint64_t, 2> hidden_weight_shape{
      hidden_size, hidden_size};
  const std::array<std::uint64_t, 2> key_value_weight_shape{
      key_value_width, hidden_size};
  const std::array<std::uint64_t, 2> intermediate_weight_shape{
      intermediate_size, hidden_size};
  const std::array<std::uint64_t, 2> down_weight_shape{
      hidden_size, intermediate_size};
  const std::array<std::uint64_t, 3> hidden_shape{
      batch, tokens, hidden_size};
  const std::array<std::uint64_t, 4> cache_shape{
      batch, maximum_context, key_value_heads, head_dim};
  const LayerWeights cpu_weights{
      const_view(input_norm, norm_shape),
      {
          const_view(query_weight, hidden_weight_shape),
          const_view(key_weight, key_value_weight_shape),
          const_view(value_weight, key_value_weight_shape),
          const_view(output_weight, hidden_weight_shape),
      },
      const_view(post_norm, norm_shape),
      {
          const_view(gate_weight, intermediate_weight_shape),
          const_view(up_weight, intermediate_weight_shape),
          const_view(down_weight, down_weight_shape),
      },
  };
  const marketforge::DecoderLayerSpec cpu_spec{
      hidden_size,
      intermediate_size,
      query_heads,
      key_value_heads,
      head_dim,
      1.0e-5F,
      marketforge::RopeSpec{10'000.0F, 16},
  };
  const auto required =
      marketforge::required_decoder_layer_workspace_floats(
          cpu_spec, batch, tokens, maximum_context);
  MF_CHECK(required);
  auto cpu_workspace = CpuWorkspace::allocate(required.value());
  MF_CHECK(cpu_workspace);
  const std::array<std::uint32_t, 1> contexts{2};
  MF_CHECK(marketforge::smollm2_decoder_layer_f32(
               cpu_weights, cpu_spec, view(expected_hidden, hidden_shape),
               CpuKvView{
                   view(expected_key_cache, cache_shape),
                   view(expected_value_cache, cache_shape),
               },
               positions, contexts, cpu_workspace.value())
               .ok());

  auto stream_result = CudaStream::create();
  auto cublas_result = CublasHandle::create();
  MF_CHECK(stream_result);
  MF_CHECK(cublas_result);
  CudaStream stream = std::move(stream_result).value();
  CublasHandle cublas = std::move(cublas_result).value();
  DeviceBuffer hidden_device = copy_to_device(hidden_half, stream);
  DeviceBuffer input_norm_device = copy_to_device(input_norm_half, stream);
  DeviceBuffer query_weight_device =
      copy_to_device(query_weight_half, stream);
  DeviceBuffer key_weight_device = copy_to_device(key_weight_half, stream);
  DeviceBuffer value_weight_device =
      copy_to_device(value_weight_half, stream);
  DeviceBuffer output_weight_device =
      copy_to_device(output_weight_half, stream);
  DeviceBuffer post_norm_device = copy_to_device(post_norm_half, stream);
  DeviceBuffer gate_weight_device =
      copy_to_device(gate_weight_half, stream);
  DeviceBuffer up_weight_device = copy_to_device(up_weight_half, stream);
  DeviceBuffer down_weight_device =
      copy_to_device(down_weight_half, stream);
  auto positions_result =
      DeviceBuffer::allocate(positions.size() * sizeof(std::uint32_t));
  auto key_cache_result = DeviceBuffer::allocate(
      expected_key_cache.size() * sizeof(__half));
  auto value_cache_result = DeviceBuffer::allocate(
      expected_value_cache.size() * sizeof(__half));
  auto normalized_result =
      DeviceBuffer::allocate(rows * hidden_size * sizeof(__half));
  auto query_result =
      DeviceBuffer::allocate(rows * hidden_size * sizeof(__half));
  auto key_result =
      DeviceBuffer::allocate(rows * key_value_width * sizeof(__half));
  auto value_result =
      DeviceBuffer::allocate(rows * key_value_width * sizeof(__half));
  auto attention_result =
      DeviceBuffer::allocate(rows * hidden_size * sizeof(__half));
  auto gate_result =
      DeviceBuffer::allocate(rows * intermediate_size * sizeof(__half));
  auto up_result =
      DeviceBuffer::allocate(rows * intermediate_size * sizeof(__half));
  MF_CHECK(positions_result);
  MF_CHECK(key_cache_result);
  MF_CHECK(value_cache_result);
  MF_CHECK(normalized_result);
  MF_CHECK(query_result);
  MF_CHECK(key_result);
  MF_CHECK(value_result);
  MF_CHECK(attention_result);
  MF_CHECK(gate_result);
  MF_CHECK(up_result);
  DeviceBuffer positions_device = std::move(positions_result).value();
  DeviceBuffer key_cache_device = std::move(key_cache_result).value();
  DeviceBuffer value_cache_device = std::move(value_cache_result).value();
  DeviceBuffer normalized_device = std::move(normalized_result).value();
  DeviceBuffer query_device = std::move(query_result).value();
  DeviceBuffer key_device = std::move(key_result).value();
  DeviceBuffer value_device = std::move(value_result).value();
  DeviceBuffer attention_device = std::move(attention_result).value();
  DeviceBuffer gate_device = std::move(gate_result).value();
  DeviceBuffer up_device = std::move(up_result).value();
  std::vector<__half> zero_cache(expected_key_cache.size(),
                                 __float2half_rn(0.0F));
  MF_CHECK(positions_device
               .copy_from_host_async(
                   positions.data(),
                   positions.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(key_cache_device
               .copy_from_host_async(
                   zero_cache.data(), zero_cache.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(value_cache_device
               .copy_from_host_async(
                   zero_cache.data(), zero_cache.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  const CudaDecoderLayerWeightsView gpu_weights{
      input_norm_device,
      {
          query_weight_device,
          key_weight_device,
          value_weight_device,
          output_weight_device,
      },
      post_norm_device,
      {
          gate_weight_device,
          up_weight_device,
          down_weight_device,
      },
  };
  const CudaDecoderLayerWorkspaceView gpu_workspace{
      normalized_device,
      query_device,
      key_device,
      value_device,
      attention_device,
      gate_device,
      up_device,
  };
  const CudaDecoderLayerSpec gpu_spec{
      hidden_size,
      intermediate_size,
      query_heads,
      key_value_heads,
      head_dim,
      1.0e-5F,
      10'000.0F,
  };
  MF_CHECK(marketforge::cuda::smollm2_decoder_layer_f16(
               gpu_weights, gpu_spec, hidden_device,
               CudaKvCacheView{key_cache_device, value_cache_device},
               positions_device, gpu_workspace, batch, tokens,
               maximum_context, cublas, stream.handle())
               .ok());
  std::vector<__half> observed_hidden(hidden_half.size());
  std::vector<__half> observed_key_cache(zero_cache.size());
  std::vector<__half> observed_value_cache(zero_cache.size());
  MF_CHECK(hidden_device
               .copy_to_host_async(
                   observed_hidden.data(),
                   observed_hidden.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(key_cache_device
               .copy_to_host_async(
                   observed_key_cache.data(),
                   observed_key_cache.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(value_cache_device
               .copy_to_host_async(
                   observed_value_cache.data(),
                   observed_value_cache.size() * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < observed_hidden.size(); ++index) {
    MF_CHECK_NEAR(
        __half2float(observed_hidden[index]), expected_hidden[index], 0.02F);
  }
  for (std::size_t index = 0; index < observed_key_cache.size(); ++index) {
    MF_CHECK_NEAR(
        __half2float(observed_key_cache[index]), expected_key_cache[index],
        0.005F);
    MF_CHECK_NEAR(
        __half2float(observed_value_cache[index]),
        expected_value_cache[index], 0.005F);
  }
}

MF_TEST(cuda_attention_f16_rejects_non_gqa_metadata) {
  auto stream_result = CudaStream::create();
  MF_CHECK(stream_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer empty;
  MF_CHECK_EQ(marketforge::cuda::attention_f16(
                  empty, empty, empty, empty, empty, 1, 1, 1, 3, 2, 2,
                  stream.handle())
                  .code,
              marketforge::ErrorCode::invalid_argument);
}

} // namespace
