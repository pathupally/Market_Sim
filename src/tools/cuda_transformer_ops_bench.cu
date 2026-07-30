#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/rope.hpp"
#include "marketforge/cuda/swiglu.hpp"

namespace {

using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

constexpr std::uint64_t query_heads = 9;
constexpr std::uint64_t key_value_heads = 3;
constexpr std::uint64_t head_dim = 64;
constexpr std::uint64_t intermediate_size = 1'536;

struct BenchmarkCase {
  std::uint64_t rows;
  std::uint32_t iterations;
};

struct Result {
  std::string_view operation;
  std::uint64_t rows;
  std::uint64_t elements;
  std::uint32_t iterations;
  double average_microseconds;
  double logical_gib_per_second;
};

[[nodiscard]] cudaStream_t
native_stream(const marketforge::cuda::StreamHandle stream) noexcept {
  return reinterpret_cast<cudaStream_t>(stream.value);
}

[[nodiscard]] bool check_cuda(const cudaError_t status,
                              const std::string_view operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

template <typename Launch>
[[nodiscard]] bool measure(const CudaStream& stream,
                           const std::uint32_t iterations, Launch launch,
                           double& average_seconds) {
  for (std::uint32_t iteration = 0; iteration < 20; ++iteration) {
    if (!launch()) {
      return false;
    }
  }
  if (!stream.synchronize().ok()) {
    return false;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
      !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
    if (start != nullptr) {
      (void)cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      (void)cudaEventDestroy(stop);
    }
    return false;
  }
  bool success =
      check_cuda(cudaEventRecord(start, native_stream(stream.handle())),
                 "cudaEventRecord(start)");
  for (std::uint32_t iteration = 0; success && iteration < iterations;
       ++iteration) {
    success = launch();
  }
  success = success &&
            check_cuda(cudaEventRecord(stop, native_stream(stream.handle())),
                       "cudaEventRecord(stop)") &&
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float elapsed_milliseconds = 0.0F;
  success = success &&
            check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start, stop),
                       "cudaEventElapsedTime");
  (void)cudaEventDestroy(start);
  (void)cudaEventDestroy(stop);
  if (!success || elapsed_milliseconds <= 0.0F) {
    return false;
  }
  average_seconds =
      static_cast<double>(elapsed_milliseconds) /
      (1'000.0 * static_cast<double>(iterations));
  return true;
}

[[nodiscard]] bool run_rope_case(const BenchmarkCase benchmark,
                                 Result& result) {
  const auto query_elements = static_cast<std::size_t>(
      benchmark.rows * query_heads * head_dim);
  const auto key_elements = static_cast<std::size_t>(
      benchmark.rows * key_value_heads * head_dim);
  std::vector<__half> query(query_elements);
  std::vector<__half> key(key_elements);
  std::vector<std::uint32_t> positions(
      static_cast<std::size_t>(benchmark.rows));
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 29) - 14) /
        32.0F);
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 23) - 11) /
        32.0F);
  }
  for (std::size_t index = 0; index < positions.size(); ++index) {
    positions[index] = static_cast<std::uint32_t>(index % 8'192);
  }

  const auto query_bytes = query_elements * sizeof(__half);
  const auto key_bytes = key_elements * sizeof(__half);
  const auto position_bytes = positions.size() * sizeof(std::uint32_t);
  auto stream_result = CudaStream::create();
  auto query_result = DeviceBuffer::allocate(query_bytes);
  auto key_result = DeviceBuffer::allocate(key_bytes);
  auto positions_result = DeviceBuffer::allocate(position_bytes);
  if (!stream_result || !query_result || !key_result || !positions_result) {
    return false;
  }
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer query_device = std::move(query_result).value();
  DeviceBuffer key_device = std::move(key_result).value();
  DeviceBuffer positions_device = std::move(positions_result).value();
  if (!query_device
           .copy_from_host_async(query.data(), query_bytes, 0, stream.handle())
           .ok() ||
      !key_device
           .copy_from_host_async(key.data(), key_bytes, 0, stream.handle())
           .ok() ||
      !positions_device
           .copy_from_host_async(positions.data(), position_bytes, 0,
                                 stream.handle())
           .ok()) {
    return false;
  }

  double average_seconds = 0.0;
  const auto launch = [&]() {
    return marketforge::cuda::apply_rope_f16(
               query_device, key_device, positions_device, 1, benchmark.rows,
               query_heads, key_value_heads, head_dim, 10'000.0F,
               stream.handle())
        .ok();
  };
  if (!measure(stream, benchmark.iterations, launch, average_seconds)) {
    return false;
  }
  const auto elements = query_elements + key_elements;
  const double logical_bytes =
      static_cast<double>(2 * (query_bytes + key_bytes) + position_bytes);
  result = {
      "rope_f16",
      benchmark.rows,
      elements,
      benchmark.iterations,
      average_seconds * 1.0e6,
      logical_bytes / average_seconds /
          static_cast<double>(UINT64_C(1) << 30),
  };
  return true;
}

[[nodiscard]] bool run_swiglu_case(const BenchmarkCase benchmark,
                                   Result& result) {
  const auto elements =
      static_cast<std::size_t>(benchmark.rows * intermediate_size);
  const auto bytes = elements * sizeof(__half);
  std::vector<__half> gate(elements);
  std::vector<__half> up(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    gate[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 31) - 15) /
        16.0F);
    up[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 19) - 9) /
        16.0F);
  }

  auto stream_result = CudaStream::create();
  auto gate_result = DeviceBuffer::allocate(bytes);
  auto up_result = DeviceBuffer::allocate(bytes);
  auto output_result = DeviceBuffer::allocate(bytes);
  if (!stream_result || !gate_result || !up_result || !output_result) {
    return false;
  }
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer gate_device = std::move(gate_result).value();
  DeviceBuffer up_device = std::move(up_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  if (!gate_device
           .copy_from_host_async(gate.data(), bytes, 0, stream.handle())
           .ok() ||
      !up_device.copy_from_host_async(up.data(), bytes, 0, stream.handle())
           .ok()) {
    return false;
  }

  double average_seconds = 0.0;
  const auto launch = [&]() {
    return marketforge::cuda::swiglu_f16(
               gate_device, up_device, output_device, elements,
               stream.handle())
        .ok();
  };
  if (!measure(stream, benchmark.iterations, launch, average_seconds)) {
    return false;
  }
  const double logical_bytes = static_cast<double>(3 * bytes);
  result = {
      "swiglu_f16",
      benchmark.rows,
      elements,
      benchmark.iterations,
      average_seconds * 1.0e6,
      logical_bytes / average_seconds /
          static_cast<double>(UINT64_C(1) << 30),
  };
  return true;
}

} // namespace

int main() {
  int device_count = 0;
  if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") ||
      device_count != 1) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                  "cudaGetDeviceProperties")) {
    return 1;
  }

  constexpr std::array<BenchmarkCase, 3> cases{{
      {1, 2'000},
      {16, 1'000},
      {256, 500},
  }};
  std::array<Result, cases.size() * 2> results{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!run_rope_case(cases[index], results[index]) ||
        !run_swiglu_case(cases[index], results[cases.size() + index])) {
      return 1;
    }
  }

  std::cout << std::fixed << std::setprecision(3)
            << "{\"schema_version\":1,\"result\":\"pass\","
               "\"operator\":\"transformer_elementwise_f16\","
               "\"model_shape\":\"SmolLM2-135M GQA 9:3x64, MLP 1536\","
               "\"gpu\":{\"name\":\""
            << properties.name << "\",\"compute_capability\":\""
            << properties.major << '.' << properties.minor
            << "\"},\"measurements\":[";
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto& result = results[index];
    std::cout << "{\"operation\":\"" << result.operation
              << "\",\"rows\":" << result.rows
              << ",\"elements\":" << result.elements
              << ",\"iterations\":" << result.iterations
              << ",\"average_microseconds\":" << result.average_microseconds
              << ",\"logical_gib_per_second\":"
              << result.logical_gib_per_second << '}';
  }
  std::cout << "]}\n";
  return 0;
}
