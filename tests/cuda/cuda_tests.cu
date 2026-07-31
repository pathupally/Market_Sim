#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "cublas_internal.hpp"
#include "cuda_internal.hpp"
#include "marketforge/core/dtype.hpp"
#include "marketforge/core/shape.hpp"
#include "marketforge/core/status.hpp"
#include "marketforge/core/tensor_view.hpp"
#include "marketforge/cpu/operators.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/greedy.hpp"
#include "marketforge/cuda/kv_cache.hpp"
#include "marketforge/cuda/linear.hpp"
#include "marketforge/cuda/rms_norm.hpp"
#include "marketforge/cuda/rope.hpp"
#include "marketforge/cuda/swiglu.hpp"
#include "marketforge/grammar/smollm2_market_actions.hpp"
#include "test_support.hpp"

namespace {

using marketforge::ErrorCode;
using marketforge::MemoryKind;
using marketforge::TensorView;
using marketforge::cuda::CublasHandle;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;
using marketforge::cuda::StreamHandle;

static_assert(!std::is_copy_constructible_v<CublasHandle>);
static_assert(!std::is_copy_assignable_v<CublasHandle>);
static_assert(std::is_nothrow_move_constructible_v<CublasHandle>);
static_assert(std::is_nothrow_move_assignable_v<CublasHandle>);
static_assert(std::is_nothrow_destructible_v<CublasHandle>);
static_assert(!std::is_copy_constructible_v<CudaStream>);
static_assert(!std::is_copy_assignable_v<CudaStream>);
static_assert(std::is_nothrow_move_constructible_v<CudaStream>);
static_assert(std::is_nothrow_move_assignable_v<CudaStream>);
static_assert(std::is_nothrow_destructible_v<CudaStream>);
static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
static_assert(!std::is_copy_assignable_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer>);
static_assert(std::is_nothrow_destructible_v<DeviceBuffer>);

template <typename T> [[nodiscard]] T&& indirect_move(T& value) noexcept {
  return static_cast<T&&>(value);
}

void run_rms_norm_parity_case(const std::uint64_t rows,
                              const std::uint64_t hidden_size,
                              const bool in_place) {
  const auto elements = static_cast<std::size_t>(rows * hidden_size);
  std::vector<float> input(elements);
  std::vector<float> weight(static_cast<std::size_t>(hidden_size));
  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto centered = static_cast<std::int32_t>(index % 37) - 18;
    input[index] = static_cast<float>(centered) * 0.0625F;
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = 0.75F + static_cast<float>(index % 17) * 0.03125F;
  }
  std::vector<float> expected(elements);
  std::vector<float> observed(elements);
  const std::array<std::uint64_t, 2> input_extents{rows, hidden_size};
  const std::array<std::uint64_t, 1> weight_extents{hidden_size};
  const auto input_shape = marketforge::make_shape(input_extents);
  const auto weight_shape = marketforge::make_shape(weight_extents);
  MF_CHECK(input_shape);
  MF_CHECK(weight_shape);
  MF_CHECK(marketforge::rms_norm_f32(
               {
                   input.data(),
                   input_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               },
               {
                   weight.data(),
                   weight_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               },
               1.0e-5F,
               TensorView{
                   expected.data(),
                   input_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               })
               .ok());

  auto stream_result = CudaStream::create();
  auto input_result = DeviceBuffer::allocate(elements * sizeof(float));
  auto weight_result = DeviceBuffer::allocate(weight.size() * sizeof(float));
  auto output_result =
      DeviceBuffer::allocate(in_place ? 0 : elements * sizeof(float));
  MF_CHECK(stream_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer input_device = std::move(input_result).value();
  DeviceBuffer weight_device = std::move(weight_result).value();
  DeviceBuffer output_device =
      in_place ? DeviceBuffer{} : std::move(output_result).value();
  MF_CHECK(input_device
               .copy_from_host_async(input.data(), elements * sizeof(float), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(weight_device
               .copy_from_host_async(weight.data(),
                                     weight.size() * sizeof(float), 0,
                                     stream.handle())
               .ok());
  DeviceBuffer& destination = in_place ? input_device : output_device;
  MF_CHECK(marketforge::cuda::rms_norm_f32(input_device, weight_device,
                                           destination, rows, hidden_size,
                                           1.0e-5F, stream.handle())
               .ok());
  MF_CHECK(destination
               .copy_to_host_async(observed.data(), elements * sizeof(float), 0,
                                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < elements; ++index) {
    MF_CHECK_NEAR(observed[index], expected[index], 3.0e-5F);
  }
}

void run_linear_parity_case(const std::uint64_t rows,
                            const std::uint64_t input_features,
                            const std::uint64_t output_features) {
  const auto input_elements =
      static_cast<std::size_t>(rows * input_features);
  const auto weight_elements =
      static_cast<std::size_t>(output_features * input_features);
  const auto output_elements =
      static_cast<std::size_t>(rows * output_features);
  std::vector<__half> input(input_elements);
  std::vector<__half> weight(weight_elements);
  std::vector<__half> observed(output_elements);
  std::vector<float> expected(output_elements, 0.0F);

  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(index % 17) - 8) /
        32.0F;
    input[index] = __float2half_rn(value);
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(index % 13) - 6) /
        64.0F;
    weight[index] = __float2half_rn(value);
  }
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t output_feature = 0;
         output_feature < output_features; ++output_feature) {
      float sum = 0.0F;
      for (std::uint64_t input_feature = 0;
           input_feature < input_features; ++input_feature) {
        const auto input_index =
            static_cast<std::size_t>(row * input_features + input_feature);
        const auto weight_index = static_cast<std::size_t>(
            output_feature * input_features + input_feature);
        sum = std::fma(__half2float(input[input_index]),
                       __half2float(weight[weight_index]), sum);
      }
      expected[static_cast<std::size_t>(row * output_features +
                                        output_feature)] = sum;
    }
  }

  auto stream_result = CudaStream::create();
  auto handle_result = CublasHandle::create();
  auto input_result =
      DeviceBuffer::allocate(input_elements * sizeof(__half));
  auto weight_result =
      DeviceBuffer::allocate(weight_elements * sizeof(__half));
  auto output_result =
      DeviceBuffer::allocate(output_elements * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(handle_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  CublasHandle handle = std::move(handle_result).value();
  DeviceBuffer input_device = std::move(input_result).value();
  DeviceBuffer weight_device = std::move(weight_result).value();
  DeviceBuffer output_device = std::move(output_result).value();

  MF_CHECK(input_device
               .copy_from_host_async(input.data(),
                                     input_elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(weight_device
               .copy_from_host_async(weight.data(),
                                     weight_elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::linear_f16(
               input_device, weight_device, output_device, rows,
               input_features, output_features, handle, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(observed.data(),
                                   output_elements * sizeof(__half), 0,
                                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < observed.size(); ++index) {
    MF_CHECK_NEAR(__half2float(observed[index]), expected[index], 0.015F);
  }
}

void run_rope_parity_case(
    const std::uint64_t batch, const std::uint64_t tokens,
    const std::uint64_t query_heads, const std::uint64_t key_value_heads,
    const std::uint64_t head_dim,
    const std::vector<std::uint32_t>& positions) {
  const auto query_elements =
      static_cast<std::size_t>(batch * tokens * query_heads * head_dim);
  const auto key_elements = static_cast<std::size_t>(
      batch * tokens * key_value_heads * head_dim);
  std::vector<__half> query(query_elements);
  std::vector<__half> key(key_elements);
  std::vector<float> expected_query(query_elements);
  std::vector<float> expected_key(key_elements);
  for (std::size_t index = 0; index < query.size(); ++index) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(index % 29) - 14) /
        32.0F;
    query[index] = __float2half_rn(value);
    expected_query[index] = __half2float(query[index]);
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    const auto value =
        static_cast<float>(static_cast<std::int32_t>(index % 23) - 11) /
        32.0F;
    key[index] = __float2half_rn(value);
    expected_key[index] = __half2float(key[index]);
  }
  const std::array<std::uint64_t, 4> query_extents{
      batch, tokens, query_heads, head_dim};
  const std::array<std::uint64_t, 4> key_extents{
      batch, tokens, key_value_heads, head_dim};
  const auto query_shape = marketforge::make_shape(query_extents);
  const auto key_shape = marketforge::make_shape(key_extents);
  MF_CHECK(query_shape);
  MF_CHECK(key_shape);
  MF_CHECK(marketforge::apply_rope_f32(
               TensorView{
                   expected_query.data(),
                   query_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               },
               TensorView{
                   expected_key.data(),
                   key_shape.value(),
                   marketforge::DType::f32,
                   MemoryKind::host,
               },
               positions, marketforge::RopeSpec{10'000.0F, 8'192})
               .ok());

  auto stream_result = CudaStream::create();
  auto query_result =
      DeviceBuffer::allocate(query_elements * sizeof(__half));
  auto key_result = DeviceBuffer::allocate(key_elements * sizeof(__half));
  auto position_result =
      DeviceBuffer::allocate(positions.size() * sizeof(std::uint32_t));
  MF_CHECK(stream_result);
  MF_CHECK(query_result);
  MF_CHECK(key_result);
  MF_CHECK(position_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer query_device = std::move(query_result).value();
  DeviceBuffer key_device = std::move(key_result).value();
  DeviceBuffer position_device = std::move(position_result).value();
  MF_CHECK(query_device
               .copy_from_host_async(query.data(),
                                     query_elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(key_device
               .copy_from_host_async(key.data(),
                                     key_elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(position_device
               .copy_from_host_async(
                   positions.data(),
                   positions.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::apply_rope_f16(
               query_device, key_device, position_device, batch, tokens,
               query_heads, key_value_heads, head_dim, 10'000.0F,
               stream.handle())
               .ok());
  MF_CHECK(query_device
               .copy_to_host_async(query.data(),
                                   query_elements * sizeof(__half), 0,
                                   stream.handle())
               .ok());
  MF_CHECK(key_device
               .copy_to_host_async(key.data(), key_elements * sizeof(__half),
                                   0, stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < query.size(); ++index) {
    MF_CHECK_NEAR(__half2float(query[index]), expected_query[index], 0.002F);
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    MF_CHECK_NEAR(__half2float(key[index]), expected_key[index], 0.002F);
  }
}

void run_swiglu_parity_case(const std::uint64_t elements,
                            const bool in_place) {
  std::vector<__half> gate(static_cast<std::size_t>(elements));
  std::vector<__half> up(static_cast<std::size_t>(elements));
  std::vector<__half> observed(static_cast<std::size_t>(elements));
  std::vector<float> expected(static_cast<std::size_t>(elements));
  for (std::size_t index = 0; index < gate.size(); ++index) {
    const auto gate_value =
        static_cast<float>(static_cast<std::int32_t>(index % 31) - 15) /
        8.0F;
    const auto up_value =
        static_cast<float>(static_cast<std::int32_t>(index % 19) - 9) /
        16.0F;
    gate[index] = __float2half_rn(gate_value);
    up[index] = __float2half_rn(up_value);
    const float rounded_gate = __half2float(gate[index]);
    expected[index] =
        rounded_gate / (1.0F + std::exp(-rounded_gate)) *
        __half2float(up[index]);
  }

  auto stream_result = CudaStream::create();
  auto gate_result = DeviceBuffer::allocate(elements * sizeof(__half));
  auto up_result = DeviceBuffer::allocate(elements * sizeof(__half));
  auto output_result =
      DeviceBuffer::allocate(in_place ? 0 : elements * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(gate_result);
  MF_CHECK(up_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer gate_device = std::move(gate_result).value();
  DeviceBuffer up_device = std::move(up_result).value();
  DeviceBuffer output_device =
      in_place ? DeviceBuffer{} : std::move(output_result).value();
  MF_CHECK(gate_device
               .copy_from_host_async(gate.data(), elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(up_device
               .copy_from_host_async(up.data(), elements * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  DeviceBuffer& destination = in_place ? gate_device : output_device;
  MF_CHECK(marketforge::cuda::swiglu_f16(
               gate_device, up_device, destination, elements, stream.handle())
               .ok());
  MF_CHECK(destination
               .copy_to_host_async(observed.data(),
                                   elements * sizeof(__half), 0,
                                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < observed.size(); ++index) {
    MF_CHECK_NEAR(__half2float(observed[index]), expected[index], 0.002F);
  }
}

void run_kv_append_parity_case(
    const std::uint64_t batch, const std::uint64_t tokens,
    const std::uint64_t maximum_context, const std::uint64_t heads,
    const std::uint64_t head_dim,
    const std::vector<std::uint32_t>& positions) {
  const auto source_elements =
      static_cast<std::size_t>(batch * tokens * heads * head_dim);
  const auto cache_elements =
      static_cast<std::size_t>(batch * maximum_context * heads * head_dim);
  std::vector<__half> key(source_elements);
  std::vector<__half> value(source_elements);
  std::vector<__half> key_cache(cache_elements, __float2half_rn(-7.0F));
  std::vector<__half> value_cache(cache_elements, __float2half_rn(-9.0F));
  auto expected_key = key_cache;
  auto expected_value = value_cache;
  for (std::size_t index = 0; index < source_elements; ++index) {
    key[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 29) - 14) /
        16.0F);
    value[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 31) - 15) /
        16.0F);
  }
  const auto token_width = static_cast<std::size_t>(heads * head_dim);
  for (std::uint64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::uint64_t token = 0; token < tokens; ++token) {
      const auto packed_token =
          static_cast<std::size_t>(batch_index * tokens + token);
      const auto source_offset = packed_token * token_width;
      const auto destination_offset =
          static_cast<std::size_t>(
              batch_index * maximum_context + positions[packed_token]) *
          token_width;
      for (std::size_t element = 0; element < token_width; ++element) {
        expected_key[destination_offset + element] =
            key[source_offset + element];
        expected_value[destination_offset + element] =
            value[source_offset + element];
      }
    }
  }

  auto stream_result = CudaStream::create();
  auto key_result = DeviceBuffer::allocate(source_elements * sizeof(__half));
  auto value_result =
      DeviceBuffer::allocate(source_elements * sizeof(__half));
  auto positions_result =
      DeviceBuffer::allocate(positions.size() * sizeof(std::uint32_t));
  auto key_cache_result =
      DeviceBuffer::allocate(cache_elements * sizeof(__half));
  auto value_cache_result =
      DeviceBuffer::allocate(cache_elements * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(key_result);
  MF_CHECK(value_result);
  MF_CHECK(positions_result);
  MF_CHECK(key_cache_result);
  MF_CHECK(value_cache_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer key_device = std::move(key_result).value();
  DeviceBuffer value_device = std::move(value_result).value();
  DeviceBuffer positions_device = std::move(positions_result).value();
  DeviceBuffer key_cache_device = std::move(key_cache_result).value();
  DeviceBuffer value_cache_device = std::move(value_cache_result).value();
  MF_CHECK(key_device
               .copy_from_host_async(
                   key.data(), source_elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(value_device
               .copy_from_host_async(
                   value.data(), source_elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(positions_device
               .copy_from_host_async(
                   positions.data(),
                   positions.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(key_cache_device
               .copy_from_host_async(
                   key_cache.data(), cache_elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(value_cache_device
               .copy_from_host_async(
                   value_cache.data(), cache_elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::append_kv_f16(
               key_device, value_device, positions_device, key_cache_device,
               value_cache_device, batch, tokens, maximum_context, heads,
               head_dim, stream.handle())
               .ok());
  MF_CHECK(key_cache_device
               .copy_to_host_async(
                   key_cache.data(), cache_elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(value_cache_device
               .copy_to_host_async(
                   value_cache.data(), cache_elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t index = 0; index < cache_elements; ++index) {
    MF_CHECK_EQ(__half2float(key_cache[index]),
                __half2float(expected_key[index]));
    MF_CHECK_EQ(__half2float(value_cache[index]),
                __half2float(expected_value[index]));
  }
}

void run_greedy_parity_case(const std::uint64_t rows,
                            const std::uint64_t vocabulary_size) {
  const auto elements = static_cast<std::size_t>(rows * vocabulary_size);
  std::vector<__half> logits(elements, __float2half_rn(-4.0F));
  std::vector<std::uint32_t> expected(static_cast<std::size_t>(rows));
  std::vector<std::uint32_t> observed(static_cast<std::size_t>(rows));
  for (std::uint64_t row = 0; row < rows; ++row) {
    const auto winner = (row * 7 + 3) % vocabulary_size;
    expected[static_cast<std::size_t>(row)] =
        static_cast<std::uint32_t>(winner);
    logits[static_cast<std::size_t>(row * vocabulary_size + winner)] =
        __float2half_rn(8.0F);
  }
  if (rows > 1 && vocabulary_size > 9) {
    logits[static_cast<std::size_t>(vocabulary_size + 9)] =
        __float2half_rn(8.0F);
    expected[1] = 9;
  }
  if (rows > 2 && vocabulary_size > 1) {
    const auto row_offset = static_cast<std::size_t>(2 * vocabulary_size);
    for (std::uint64_t token = 0; token < vocabulary_size; ++token) {
      logits[row_offset + static_cast<std::size_t>(token)] =
          __float2half_rn(-std::numeric_limits<float>::infinity());
    }
    logits[row_offset] =
        __float2half_rn(std::numeric_limits<float>::quiet_NaN());
    expected[2] = 1;
  }

  auto stream_result = CudaStream::create();
  auto logits_result = DeviceBuffer::allocate(elements * sizeof(__half));
  auto tokens_result =
      DeviceBuffer::allocate(rows * sizeof(std::uint32_t));
  MF_CHECK(stream_result);
  MF_CHECK(logits_result);
  MF_CHECK(tokens_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer logits_device = std::move(logits_result).value();
  DeviceBuffer tokens_device = std::move(tokens_result).value();
  MF_CHECK(logits_device
               .copy_from_host_async(
                   logits.data(), elements * sizeof(__half), 0,
                   stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::greedy_select_f16(
               logits_device, tokens_device, rows, vocabulary_size,
               stream.handle())
               .ok());
  MF_CHECK(tokens_device
               .copy_to_host_async(
                   observed.data(), rows * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (std::size_t row = 0; row < observed.size(); ++row) {
    MF_CHECK_EQ(observed[row], expected[row]);
  }
}

void run_restricted_greedy_dfa_parity_case() {
  auto catalog_result = marketforge::SmolLm2MarketActionCatalog::create();
  MF_CHECK(catalog_result);
  auto catalog = std::move(catalog_result).value();
  const auto& dfa = catalog.dfa();

  std::vector<marketforge::GrammarState> states;
  std::vector<std::vector<marketforge::token_id_t>> allowed_by_row;
  std::size_t maximum_allowed_tokens = 0;
  for (std::uint32_t state = 0;
       state < dfa.state_count() && states.size() < 8; ++state) {
    const auto allowed = dfa.allowed(marketforge::GrammarState{state});
    MF_CHECK(allowed);
    if (allowed.value().empty()) {
      continue;
    }
    states.push_back(marketforge::GrammarState{state});
    std::vector<marketforge::token_id_t> tokens;
    for (const auto& arc : allowed.value()) {
      tokens.push_back(arc.token);
    }
    maximum_allowed_tokens =
        std::max(maximum_allowed_tokens, tokens.size());
    allowed_by_row.push_back(std::move(tokens));
  }
  MF_CHECK(!states.empty());

  const auto rows = states.size();
  const auto vocabulary_size =
      marketforge::SmolLm2MarketActionCatalog::vocabulary_size();
  const auto logits_elements = rows * vocabulary_size;
  const auto allowed_elements = rows * maximum_allowed_tokens;
  std::vector<__half> logits(logits_elements, __float2half_rn(-8.0F));
  std::vector<std::uint32_t> allowed_tokens(
      allowed_elements,
      marketforge::cuda::restricted_greedy_invalid_token_id);
  std::vector<std::uint32_t> allowed_counts(rows);
  std::vector<std::uint32_t> expected(rows);
  std::vector<std::uint32_t> observed(rows);

  for (std::size_t row = 0; row < rows; ++row) {
    const auto& candidates = allowed_by_row[row];
    allowed_counts[row] = static_cast<std::uint32_t>(candidates.size());
    auto disallowed = vocabulary_size - 1U;
    while (std::find(candidates.begin(), candidates.end(), disallowed) !=
           candidates.end()) {
      --disallowed;
    }
    logits[row * vocabulary_size + disallowed] =
        __float2half_rn(100.0F);

    auto winner = marketforge::cuda::restricted_greedy_invalid_token_id;
    float maximum = -std::numeric_limits<float>::infinity();
    auto fallback = marketforge::cuda::restricted_greedy_invalid_token_id;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const auto token = candidates[index];
      allowed_tokens[row * maximum_allowed_tokens + index] = token;
      const float value =
          row == 0
              ? 4.0F
              : (row == 1
                     ? std::numeric_limits<float>::quiet_NaN()
                     : static_cast<float>((token * 17U + row) % 31U));
      logits[row * vocabulary_size + token] = __float2half_rn(value);
      const float rounded =
          __half2float(logits[row * vocabulary_size + token]);
      fallback = std::min(fallback, token);
      if (rounded > maximum ||
          (rounded == maximum && token < winner)) {
        maximum = rounded;
        winner = token;
      }
    }
    expected[row] =
        winner == marketforge::cuda::restricted_greedy_invalid_token_id
            ? fallback
            : winner;
  }

  auto stream_result = CudaStream::create();
  auto logits_result =
      DeviceBuffer::allocate(logits.size() * sizeof(__half));
  auto allowed_result =
      DeviceBuffer::allocate(allowed_tokens.size() *
                             sizeof(std::uint32_t));
  auto counts_result =
      DeviceBuffer::allocate(allowed_counts.size() *
                             sizeof(std::uint32_t));
  auto output_result =
      DeviceBuffer::allocate(observed.size() * sizeof(std::uint32_t));
  MF_CHECK(stream_result);
  MF_CHECK(logits_result);
  MF_CHECK(allowed_result);
  MF_CHECK(counts_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer logits_device = std::move(logits_result).value();
  DeviceBuffer allowed_device = std::move(allowed_result).value();
  DeviceBuffer counts_device = std::move(counts_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  MF_CHECK(logits_device
               .copy_from_host_async(logits.data(),
                                     logits.size() * sizeof(__half), 0,
                                     stream.handle())
               .ok());
  MF_CHECK(allowed_device
               .copy_from_host_async(
                   allowed_tokens.data(),
                   allowed_tokens.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(counts_device
               .copy_from_host_async(
                   allowed_counts.data(),
                   allowed_counts.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::restricted_greedy_select_f16(
               logits_device, allowed_device, counts_device, output_device,
               rows, vocabulary_size, maximum_allowed_tokens,
               stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(),
                   observed.size() * sizeof(std::uint32_t), 0,
                   stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  MF_CHECK_EQ(observed, expected);
}

void run_restricted_output_head_parity_case() {
  constexpr std::uint64_t rows = 3;
  constexpr std::uint64_t hidden_size = 8;
  constexpr std::uint64_t vocabulary_size = 17;
  constexpr std::uint64_t maximum_allowed = 5;
  std::array<__half, rows * hidden_size> hidden{};
  std::array<__half, vocabulary_size * hidden_size> embedding{};
  std::array<std::uint32_t, rows * maximum_allowed> allowed{
      9, 3, 14, 0, 0,
      6, 4, 8, 0, 0,
      7, 1, 0, 0, 0,
  };
  std::array<std::uint32_t, rows> counts{3, 3, 2};
  std::array<std::uint32_t, rows> expected{};
  std::array<std::uint32_t, rows> observed{};

  for (std::size_t index = 0; index < hidden.size(); ++index) {
    const auto centered = static_cast<std::int32_t>(index % 11U) - 5;
    hidden[index] = __float2half_rn(static_cast<float>(centered) / 8.0F);
  }
  for (std::size_t index = 0; index < embedding.size(); ++index) {
    const auto centered = static_cast<std::int32_t>((index * 7U) % 19U) - 9;
    embedding[index] =
        __float2half_rn(static_cast<float>(centered) / 16.0F);
  }
  for (std::uint64_t feature = 0; feature < hidden_size; ++feature) {
    embedding[6 * hidden_size + feature] =
        embedding[4 * hidden_size + feature];
    hidden[2 * hidden_size + feature] =
        __float2half_rn(std::numeric_limits<float>::quiet_NaN());
  }

  for (std::size_t row = 0; row < rows; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    auto winner = marketforge::cuda::restricted_greedy_invalid_token_id;
    auto fallback = marketforge::cuda::restricted_greedy_invalid_token_id;
    for (std::size_t index = 0; index < counts[row]; ++index) {
      const auto token = allowed[row * maximum_allowed + index];
      fallback = std::min(fallback, token);
      float score = 0.0F;
      for (std::size_t feature = 0; feature < hidden_size; ++feature) {
        score = std::fma(
            __half2float(hidden[row * hidden_size + feature]),
            __half2float(embedding[token * hidden_size + feature]), score);
      }
      if (score > maximum || (score == maximum && token < winner)) {
        maximum = score;
        winner = token;
      }
    }
    expected[row] =
        winner == marketforge::cuda::restricted_greedy_invalid_token_id
            ? fallback
            : winner;
  }

  auto stream_result = CudaStream::create();
  auto hidden_result = DeviceBuffer::allocate(sizeof(hidden));
  auto embedding_result = DeviceBuffer::allocate(sizeof(embedding));
  auto allowed_result = DeviceBuffer::allocate(sizeof(allowed));
  auto counts_result = DeviceBuffer::allocate(sizeof(counts));
  auto output_result = DeviceBuffer::allocate(sizeof(observed));
  MF_CHECK(stream_result);
  MF_CHECK(hidden_result);
  MF_CHECK(embedding_result);
  MF_CHECK(allowed_result);
  MF_CHECK(counts_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer hidden_device = std::move(hidden_result).value();
  DeviceBuffer embedding_device = std::move(embedding_result).value();
  DeviceBuffer allowed_device = std::move(allowed_result).value();
  DeviceBuffer counts_device = std::move(counts_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  MF_CHECK(hidden_device
               .copy_from_host_async(
                   hidden.data(), sizeof(hidden), 0, stream.handle())
               .ok());
  MF_CHECK(embedding_device
               .copy_from_host_async(
                   embedding.data(), sizeof(embedding), 0, stream.handle())
               .ok());
  MF_CHECK(allowed_device
               .copy_from_host_async(
                   allowed.data(), sizeof(allowed), 0, stream.handle())
               .ok());
  MF_CHECK(counts_device
               .copy_from_host_async(
                   counts.data(), sizeof(counts), 0, stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::restricted_output_head_f16(
               hidden_device, embedding_device, allowed_device,
               counts_device, output_device, rows, hidden_size,
               vocabulary_size, maximum_allowed, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(), sizeof(observed), 0, stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  MF_CHECK_EQ(observed, expected);
}

MF_TEST(cuda_error_classification_preserves_numeric_detail) {
  const auto runtime =
      marketforge::cuda::detail::runtime_status(cudaErrorInvalidValue);
  MF_CHECK_EQ(runtime.code, ErrorCode::cuda_runtime_failure);
  MF_CHECK_EQ(runtime.detail,
              static_cast<std::uint32_t>(cudaErrorInvalidValue));

  const auto launch =
      marketforge::cuda::detail::launch_status(cudaErrorInvalidConfiguration);
  MF_CHECK_EQ(launch.code, ErrorCode::cuda_backend_failure);
  MF_CHECK_EQ(launch.detail,
              static_cast<std::uint32_t>(cudaErrorInvalidConfiguration));

  const auto cublas =
      marketforge::cuda::detail::cublas_status(CUBLAS_STATUS_INVALID_VALUE);
  MF_CHECK_EQ(cublas.code, ErrorCode::cuda_backend_failure);
  MF_CHECK_EQ(cublas.detail,
              static_cast<std::uint32_t>(CUBLAS_STATUS_INVALID_VALUE));
}

MF_TEST(cuda_empty_states_and_zero_copy_are_safe) {
  CudaStream empty_stream;
  MF_CHECK(!empty_stream.handle().valid());
  MF_CHECK_EQ(empty_stream.synchronize().code, ErrorCode::invalid_argument);

  auto stream_result = CudaStream::create();
  MF_CHECK(stream_result);
  CudaStream stream = std::move(stream_result).value();
  auto zero_result = DeviceBuffer::allocate(0);
  MF_CHECK(zero_result);
  DeviceBuffer zero = std::move(zero_result).value();
  MF_CHECK_EQ(zero.size_bytes(), 0);
  MF_CHECK(!zero.address().valid());
  MF_CHECK(zero.copy_from_host_async(nullptr, 0, 0, stream.handle()).ok());
  MF_CHECK(zero.copy_to_host_async(nullptr, 0, 0, stream.handle()).ok());
  MF_CHECK_EQ(zero.copy_from_host_async(nullptr, 0, 0, {}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(zero.copy_from_host_async(nullptr, 0, 1, stream.handle()).code,
              ErrorCode::invalid_argument);
}

MF_TEST(cuda_move_and_copy_validation) {
  auto stream_result = CudaStream::create();
  MF_CHECK(stream_result);
  CudaStream source_stream = std::move(stream_result).value();
  auto live_stream_result = CudaStream::create();
  MF_CHECK(live_stream_result);
  CudaStream stream = std::move(live_stream_result).value();
  stream = std::move(source_stream);
  MF_CHECK(!source_stream.handle().valid());
  stream = indirect_move(stream);
  MF_CHECK(stream.handle().valid());

  auto first_result = DeviceBuffer::allocate(16);
  auto second_result = DeviceBuffer::allocate(32);
  MF_CHECK(first_result);
  MF_CHECK(second_result);
  DeviceBuffer first = std::move(first_result).value();
  DeviceBuffer second = std::move(second_result).value();
  second = std::move(first);
  MF_CHECK(!first.address().valid());
  MF_CHECK_EQ(first.size_bytes(), 0);
  MF_CHECK_EQ(second.size_bytes(), 16);
  second = indirect_move(second);
  MF_CHECK_EQ(second.size_bytes(), 16);

  std::uint32_t value = 0;
  MF_CHECK_EQ(
      second.copy_from_host_async(nullptr, sizeof(value), 0, stream.handle())
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      second.copy_from_host_async(&value, sizeof(value), 0, StreamHandle{})
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(second
                  .copy_from_host_async(
                      &value, 2, std::numeric_limits<std::uint64_t>::max(),
                      stream.handle())
                  .code,
              ErrorCode::arithmetic_overflow);
  MF_CHECK_EQ(
      second.copy_from_host_async(&value, sizeof(value), 15, stream.handle())
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      second.copy_to_host_async(nullptr, sizeof(value), 0, stream.handle())
          .code,
      ErrorCode::invalid_argument);

  auto first_handle_result = CublasHandle::create();
  auto second_handle_result = CublasHandle::create();
  MF_CHECK(first_handle_result);
  MF_CHECK(second_handle_result);
  CublasHandle first_handle = std::move(first_handle_result).value();
  CublasHandle second_handle = std::move(second_handle_result).value();
  second_handle = std::move(first_handle);
  MF_CHECK(!first_handle.handle().valid());
  MF_CHECK(second_handle.handle().valid());
  second_handle = indirect_move(second_handle);
  MF_CHECK(second_handle.handle().valid());
}

MF_TEST(cuda_rms_norm_matches_cpu_for_smollm2_and_boundary_shapes) {
  run_rms_norm_parity_case(2, 1, false);
  run_rms_norm_parity_case(3, 255, false);
  run_rms_norm_parity_case(3, 256, false);
  run_rms_norm_parity_case(3, 257, false);
  run_rms_norm_parity_case(7, 576, false);
  run_rms_norm_parity_case(2, 1024, true);
}

MF_TEST(cuda_rms_norm_rejects_invalid_metadata_before_launch) {
  auto stream_result = CudaStream::create();
  auto input_result = DeviceBuffer::allocate(8 * sizeof(float));
  auto weight_result = DeviceBuffer::allocate(4 * sizeof(float));
  auto output_result = DeviceBuffer::allocate(8 * sizeof(float));
  MF_CHECK(stream_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer input = std::move(input_result).value();
  DeviceBuffer weight = std::move(weight_result).value();
  DeviceBuffer output = std::move(output_result).value();

  MF_CHECK_EQ(marketforge::cuda::rms_norm_f32(input, weight, output, 0, 4,
                                              1.0e-5F, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::rms_norm_f32(
                  input, weight, output, 2, 4,
                  std::numeric_limits<float>::quiet_NaN(), stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::rms_norm_f32(input, weight, output, 2, 4, 1.0e-5F, {})
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::rms_norm_f32(input, weight, output, 1, 4,
                                              1.0e-5F, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
}

MF_TEST(cuda_linear_f16_matches_reference_for_tiny_and_smollm2_shapes) {
  run_linear_parity_case(2, 3, 2);
  run_linear_parity_case(3, 576, 1'536);
}

MF_TEST(cuda_linear_f16_rejects_invalid_metadata_before_cublas) {
  auto stream_result = CudaStream::create();
  auto handle_result = CublasHandle::create();
  auto input_result = DeviceBuffer::allocate(6 * sizeof(__half));
  auto weight_result = DeviceBuffer::allocate(6 * sizeof(__half));
  auto output_result = DeviceBuffer::allocate(4 * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(handle_result);
  MF_CHECK(input_result);
  MF_CHECK(weight_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  CublasHandle handle = std::move(handle_result).value();
  DeviceBuffer input = std::move(input_result).value();
  DeviceBuffer weight = std::move(weight_result).value();
  DeviceBuffer output = std::move(output_result).value();

  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 0, 3, 2, handle, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 2, 3, 2, handle, {})
                  .code,
              ErrorCode::invalid_argument);
  CublasHandle empty_handle;
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 2, 3, 2, empty_handle,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, output, 1, 3, 2, handle, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::linear_f16(
                  input, weight, input, 2, 3, 2, handle, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::linear_f16(
          input, weight, output, std::numeric_limits<std::uint64_t>::max(), 2,
          2, handle, stream.handle())
          .code,
      ErrorCode::arithmetic_overflow);
}

MF_TEST(cuda_rope_f16_matches_llama_half_rotation_for_smollm2_gqa) {
  run_rope_parity_case(1, 2, 1, 1, 4, {0, 1});
  run_rope_parity_case(2, 3, 9, 3, 64, {0, 1, 2, 32, 511, 8'191});
}

MF_TEST(cuda_rope_f16_rejects_invalid_metadata_before_launch) {
  auto stream_result = CudaStream::create();
  auto query_result = DeviceBuffer::allocate(2 * 2 * sizeof(__half));
  auto key_result = DeviceBuffer::allocate(2 * 2 * sizeof(__half));
  auto position_result = DeviceBuffer::allocate(2 * sizeof(std::uint32_t));
  MF_CHECK(stream_result);
  MF_CHECK(query_result);
  MF_CHECK(key_result);
  MF_CHECK(position_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer query = std::move(query_result).value();
  DeviceBuffer key = std::move(key_result).value();
  DeviceBuffer positions = std::move(position_result).value();

  MF_CHECK_EQ(marketforge::cuda::apply_rope_f16(
                  query, key, positions, 1, 2, 1, 1, 2, 10'000.0F, {})
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::apply_rope_f16(
                  query, key, positions, 1, 2, 1, 1, 3, 10'000.0F,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::apply_rope_f16(
                  query, key, positions, 1, 1, 1, 1, 2, 10'000.0F,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::apply_rope_f16(
                  query, query, positions, 1, 2, 1, 1, 2, 10'000.0F,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::apply_rope_f16(
                  query, key, positions,
                  std::numeric_limits<std::uint64_t>::max(), 2, 1, 1, 2,
                  10'000.0F, stream.handle())
                  .code,
              ErrorCode::arithmetic_overflow);
}

MF_TEST(cuda_swiglu_f16_matches_reference_in_place_and_out_of_place) {
  run_swiglu_parity_case(1, false);
  run_swiglu_parity_case(3 * 1'536, false);
  run_swiglu_parity_case(3 * 1'536, true);
}

MF_TEST(cuda_swiglu_f16_rejects_invalid_metadata_before_launch) {
  auto stream_result = CudaStream::create();
  auto gate_result = DeviceBuffer::allocate(4 * sizeof(__half));
  auto up_result = DeviceBuffer::allocate(4 * sizeof(__half));
  auto output_result = DeviceBuffer::allocate(4 * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(gate_result);
  MF_CHECK(up_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer gate = std::move(gate_result).value();
  DeviceBuffer up = std::move(up_result).value();
  DeviceBuffer output = std::move(output_result).value();

  MF_CHECK_EQ(
      marketforge::cuda::swiglu_f16(gate, up, output, 0, stream.handle()).code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::swiglu_f16(gate, up, output, 4, {}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::swiglu_f16(gate, up, output, 3, stream.handle()).code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::swiglu_f16(gate, gate, output, 4, stream.handle())
          .code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::swiglu_f16(gate, up, up, 4, stream.handle()).code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::swiglu_f16(
          gate, up, output, std::numeric_limits<std::uint64_t>::max(),
          stream.handle())
          .code,
      ErrorCode::arithmetic_overflow);
}

MF_TEST(cuda_kv_append_f16_changes_only_selected_smollm2_cache_slots) {
  run_kv_append_parity_case(1, 2, 8, 1, 4, {0, 7});
  run_kv_append_parity_case(2, 3, 16, 3, 64, {0, 5, 15, 2, 8, 14});
}

MF_TEST(cuda_kv_append_f16_rejects_invalid_metadata_before_launch) {
  auto stream_result = CudaStream::create();
  auto source_result = DeviceBuffer::allocate(4 * sizeof(__half));
  auto value_result = DeviceBuffer::allocate(4 * sizeof(__half));
  auto positions_result =
      DeviceBuffer::allocate(2 * sizeof(std::uint32_t));
  auto key_cache_result = DeviceBuffer::allocate(8 * sizeof(__half));
  auto value_cache_result = DeviceBuffer::allocate(8 * sizeof(__half));
  MF_CHECK(stream_result);
  MF_CHECK(source_result);
  MF_CHECK(value_result);
  MF_CHECK(positions_result);
  MF_CHECK(key_cache_result);
  MF_CHECK(value_cache_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer key = std::move(source_result).value();
  DeviceBuffer value = std::move(value_result).value();
  DeviceBuffer positions = std::move(positions_result).value();
  DeviceBuffer key_cache = std::move(key_cache_result).value();
  DeviceBuffer value_cache = std::move(value_cache_result).value();

  MF_CHECK_EQ(marketforge::cuda::append_kv_f16(
                  key, value, positions, key_cache, value_cache, 1, 2, 4, 1,
                  2, {})
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::append_kv_f16(
                  key, value, positions, key_cache, value_cache, 1, 1, 4, 1,
                  2, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::append_kv_f16(
                  key, value, positions, key_cache, value_cache,
                  std::numeric_limits<std::uint64_t>::max(), 2, 4, 1, 2,
                  stream.handle())
                  .code,
              ErrorCode::arithmetic_overflow);
}

MF_TEST(cuda_greedy_f16_matches_lowest_id_tie_rule_at_smollm2_vocab) {
  run_greedy_parity_case(1, 17);
  run_greedy_parity_case(3, 49'152);
}

MF_TEST(cuda_greedy_f16_rejects_invalid_metadata_before_launch) {
  auto stream_result = CudaStream::create();
  auto logits_result = DeviceBuffer::allocate(8 * sizeof(__half));
  auto tokens_result = DeviceBuffer::allocate(sizeof(std::uint32_t));
  MF_CHECK(stream_result);
  MF_CHECK(logits_result);
  MF_CHECK(tokens_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer logits = std::move(logits_result).value();
  DeviceBuffer tokens = std::move(tokens_result).value();

  MF_CHECK_EQ(marketforge::cuda::greedy_select_f16(
                  logits, tokens, 0, 8, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      marketforge::cuda::greedy_select_f16(logits, tokens, 1, 8, {}).code,
      ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::greedy_select_f16(
                  logits, tokens, 1, 7, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::greedy_select_f16(
                  logits, tokens, 1,
                  static_cast<std::uint64_t>(
                      std::numeric_limits<std::uint32_t>::max()) +
                      1,
                  stream.handle())
                  .code,
              ErrorCode::resource_limit);
}

MF_TEST(cuda_restricted_greedy_f16_matches_market_action_dfa) {
  run_restricted_greedy_dfa_parity_case();
}

MF_TEST(cuda_restricted_greedy_f16_defines_invalid_device_rows) {
  constexpr std::uint64_t rows = 1;
  constexpr std::uint64_t vocabulary_size = 8;
  constexpr std::uint64_t maximum_allowed = 4;
  std::array<__half, vocabulary_size> logits{};
  std::array<std::uint32_t, maximum_allowed> allowed{1, 3, 5, 7};
  std::array<std::uint32_t, rows> counts{0};
  std::array<std::uint32_t, rows> observed{};

  auto stream_result = CudaStream::create();
  auto logits_result = DeviceBuffer::allocate(sizeof(logits));
  auto allowed_result = DeviceBuffer::allocate(sizeof(allowed));
  auto counts_result = DeviceBuffer::allocate(sizeof(counts));
  auto output_result = DeviceBuffer::allocate(sizeof(observed));
  MF_CHECK(stream_result);
  MF_CHECK(logits_result);
  MF_CHECK(allowed_result);
  MF_CHECK(counts_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer logits_device = std::move(logits_result).value();
  DeviceBuffer allowed_device = std::move(allowed_result).value();
  DeviceBuffer counts_device = std::move(counts_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  MF_CHECK(logits_device
               .copy_from_host_async(
                   logits.data(), sizeof(logits), 0, stream.handle())
               .ok());
  MF_CHECK(allowed_device
               .copy_from_host_async(
                   allowed.data(), sizeof(allowed), 0, stream.handle())
               .ok());
  MF_CHECK(counts_device
               .copy_from_host_async(
                   counts.data(), sizeof(counts), 0, stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::restricted_greedy_select_f16(
               logits_device, allowed_device, counts_device, output_device,
               rows, vocabulary_size, maximum_allowed, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(), sizeof(observed), 0, stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  MF_CHECK_EQ(observed[0],
              marketforge::cuda::restricted_greedy_invalid_token_id);

  counts[0] = static_cast<std::uint32_t>(maximum_allowed + 1U);
  MF_CHECK(counts_device
               .copy_from_host_async(
                   counts.data(), sizeof(counts), 0, stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::restricted_greedy_select_f16(
               logits_device, allowed_device, counts_device, output_device,
               rows, vocabulary_size, maximum_allowed, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(), sizeof(observed), 0, stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  MF_CHECK_EQ(observed[0],
              marketforge::cuda::restricted_greedy_invalid_token_id);

  allowed[1] = static_cast<std::uint32_t>(vocabulary_size);
  counts[0] = 2;
  MF_CHECK(allowed_device
               .copy_from_host_async(
                   allowed.data(), sizeof(allowed), 0, stream.handle())
               .ok());
  MF_CHECK(counts_device
               .copy_from_host_async(
                   counts.data(), sizeof(counts), 0, stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::restricted_greedy_select_f16(
               logits_device, allowed_device, counts_device, output_device,
               rows, vocabulary_size, maximum_allowed, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(), sizeof(observed), 0, stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  MF_CHECK_EQ(observed[0],
              marketforge::cuda::restricted_greedy_invalid_token_id);

  MF_CHECK_EQ(marketforge::cuda::restricted_greedy_select_f16(
                  logits_device, allowed_device, counts_device,
                  output_device, rows, vocabulary_size, 0, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::restricted_greedy_select_f16(
                  logits_device, allowed_device, counts_device,
                  output_device, rows, vocabulary_size,
                  maximum_allowed - 1U, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::restricted_greedy_select_f16(
                  logits_device, allowed_device, counts_device,
                  counts_device, rows, vocabulary_size, maximum_allowed,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
}

MF_TEST(cuda_restricted_output_head_f16_matches_fp32_reference) {
  run_restricted_output_head_parity_case();
}

MF_TEST(cuda_restricted_output_head_f16_rejects_invalid_metadata) {
  constexpr std::uint64_t rows = 1;
  constexpr std::uint64_t hidden_size = 4;
  constexpr std::uint64_t vocabulary_size = 8;
  constexpr std::uint64_t maximum_allowed = 2;
  auto stream_result = CudaStream::create();
  auto hidden_result =
      DeviceBuffer::allocate(rows * hidden_size * sizeof(__half));
  auto embedding_result =
      DeviceBuffer::allocate(vocabulary_size * hidden_size * sizeof(__half));
  auto allowed_result = DeviceBuffer::allocate(
      rows * maximum_allowed * sizeof(std::uint32_t));
  auto counts_result =
      DeviceBuffer::allocate(rows * sizeof(std::uint32_t));
  auto output_result =
      DeviceBuffer::allocate(rows * sizeof(std::uint32_t));
  MF_CHECK(stream_result);
  MF_CHECK(hidden_result);
  MF_CHECK(embedding_result);
  MF_CHECK(allowed_result);
  MF_CHECK(counts_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer hidden = std::move(hidden_result).value();
  DeviceBuffer embedding = std::move(embedding_result).value();
  DeviceBuffer allowed = std::move(allowed_result).value();
  DeviceBuffer counts = std::move(counts_result).value();
  DeviceBuffer output = std::move(output_result).value();

  MF_CHECK_EQ(marketforge::cuda::restricted_output_head_f16(
                  hidden, embedding, allowed, counts, output, 0, hidden_size,
                  vocabulary_size, maximum_allowed, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::restricted_output_head_f16(
                  hidden, embedding, allowed, counts, output, rows,
                  hidden_size, vocabulary_size, 0, stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::restricted_output_head_f16(
                  hidden, embedding, allowed, counts, output, rows,
                  hidden_size, vocabulary_size, maximum_allowed, {})
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::restricted_output_head_f16(
                  hidden, embedding, allowed, counts, counts, rows,
                  hidden_size, vocabulary_size, maximum_allowed,
                  stream.handle())
                  .code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(marketforge::cuda::restricted_output_head_f16(
                  hidden, embedding, allowed, counts, output,
                  std::numeric_limits<std::uint64_t>::max(), hidden_size,
                  vocabulary_size, maximum_allowed, stream.handle())
                  .code,
              ErrorCode::resource_limit);
}

MF_TEST(cuda_restricted_output_head_f16_defines_invalid_device_rows) {
  constexpr std::uint64_t rows = 3;
  constexpr std::uint64_t hidden_size = 2;
  constexpr std::uint64_t vocabulary_size = 5;
  constexpr std::uint64_t maximum_allowed = 2;
  std::array<__half, rows * hidden_size> hidden{};
  std::array<__half, vocabulary_size * hidden_size> embedding{};
  std::array<std::uint32_t, rows * maximum_allowed> allowed{
      1, 2,
      2, 3,
      1, 5,
  };
  std::array<std::uint32_t, rows> counts{0, 3, 2};
  std::array<std::uint32_t, rows> observed{};
  auto stream_result = CudaStream::create();
  auto hidden_result = DeviceBuffer::allocate(sizeof(hidden));
  auto embedding_result = DeviceBuffer::allocate(sizeof(embedding));
  auto allowed_result = DeviceBuffer::allocate(sizeof(allowed));
  auto counts_result = DeviceBuffer::allocate(sizeof(counts));
  auto output_result = DeviceBuffer::allocate(sizeof(observed));
  MF_CHECK(stream_result);
  MF_CHECK(hidden_result);
  MF_CHECK(embedding_result);
  MF_CHECK(allowed_result);
  MF_CHECK(counts_result);
  MF_CHECK(output_result);
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer hidden_device = std::move(hidden_result).value();
  DeviceBuffer embedding_device = std::move(embedding_result).value();
  DeviceBuffer allowed_device = std::move(allowed_result).value();
  DeviceBuffer counts_device = std::move(counts_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  MF_CHECK(hidden_device
               .copy_from_host_async(
                   hidden.data(), sizeof(hidden), 0, stream.handle())
               .ok());
  MF_CHECK(embedding_device
               .copy_from_host_async(
                   embedding.data(), sizeof(embedding), 0, stream.handle())
               .ok());
  MF_CHECK(allowed_device
               .copy_from_host_async(
                   allowed.data(), sizeof(allowed), 0, stream.handle())
               .ok());
  MF_CHECK(counts_device
               .copy_from_host_async(
                   counts.data(), sizeof(counts), 0, stream.handle())
               .ok());
  MF_CHECK(marketforge::cuda::restricted_output_head_f16(
               hidden_device, embedding_device, allowed_device,
               counts_device, output_device, rows, hidden_size,
               vocabulary_size, maximum_allowed, stream.handle())
               .ok());
  MF_CHECK(output_device
               .copy_to_host_async(
                   observed.data(), sizeof(observed), 0, stream.handle())
               .ok());
  MF_CHECK(stream.synchronize().ok());
  for (const auto token : observed) {
    MF_CHECK_EQ(token,
                marketforge::cuda::restricted_greedy_invalid_token_id);
  }
}

} // namespace
