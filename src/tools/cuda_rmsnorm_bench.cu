#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/rms_norm.hpp"

namespace {

using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

struct BenchmarkCase {
  std::uint64_t rows;
  std::uint32_t iterations;
};

struct Result {
  std::uint64_t rows;
  std::uint64_t hidden_size;
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

[[nodiscard]] bool run_case(const BenchmarkCase benchmark,
                            const std::uint64_t hidden_size, Result& result) {
  const auto elements = static_cast<std::size_t>(benchmark.rows * hidden_size);
  const auto tensor_bytes = elements * sizeof(float);
  const auto weight_bytes =
      static_cast<std::size_t>(hidden_size) * sizeof(float);
  std::vector<float> input(elements);
  std::vector<float> weight(static_cast<std::size_t>(hidden_size));
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<std::int32_t>(index % 29) - 14) *
        0.0625F;
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = 0.75F + static_cast<float>(index % 13) * 0.03125F;
  }

  auto stream_result = CudaStream::create();
  auto input_result = DeviceBuffer::allocate(tensor_bytes);
  auto weight_result = DeviceBuffer::allocate(weight_bytes);
  auto output_result = DeviceBuffer::allocate(tensor_bytes);
  if (!stream_result || !input_result || !weight_result || !output_result) {
    return false;
  }
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer input_device = std::move(input_result).value();
  DeviceBuffer weight_device = std::move(weight_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  if (!input_device
           .copy_from_host_async(input.data(), tensor_bytes, 0, stream.handle())
           .ok() ||
      !weight_device
           .copy_from_host_async(weight.data(), weight_bytes, 0,
                                 stream.handle())
           .ok()) {
    return false;
  }
  for (std::uint32_t iteration = 0; iteration < 20; ++iteration) {
    if (!marketforge::cuda::rms_norm_f32(input_device, weight_device,
                                         output_device, benchmark.rows,
                                         hidden_size, 1.0e-5F, stream.handle())
             .ok()) {
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
  for (std::uint32_t iteration = 0; success && iteration < benchmark.iterations;
       ++iteration) {
    success = marketforge::cuda::rms_norm_f32(
                  input_device, weight_device, output_device, benchmark.rows,
                  hidden_size, 1.0e-5F, stream.handle())
                  .ok();
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

  const double average_seconds =
      (static_cast<double>(elapsed_milliseconds) / 1'000.0) /
      static_cast<double>(benchmark.iterations);
  const double logical_bytes =
      static_cast<double>(tensor_bytes * 2 + weight_bytes);
  result = {
      benchmark.rows,
      hidden_size,
      benchmark.iterations,
      average_seconds * 1.0e6,
      logical_bytes / average_seconds / static_cast<double>(UINT64_C(1) << 30),
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

  constexpr std::uint64_t hidden_size = 576;
  constexpr std::array<BenchmarkCase, 4> cases{{
      {1, 2'000},
      {16, 1'000},
      {256, 500},
      {1'024, 200},
  }};
  std::array<Result, cases.size()> results{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!run_case(cases[index], hidden_size, results[index])) {
      return 1;
    }
  }

  std::cout << std::fixed << std::setprecision(3)
            << "{\"schema_version\":1,\"result\":\"pass\","
               "\"operator\":\"rms_norm_f32\","
               "\"model_shape\":\"SmolLM2-135M hidden_size=576\","
               "\"gpu\":{\"name\":\""
            << properties.name << "\",\"compute_capability\":\""
            << properties.major << '.' << properties.minor
            << "\"},\"measurements\":[";
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto& result = results[index];
    std::cout << "{\"rows\":" << result.rows
              << ",\"hidden_size\":" << result.hidden_size
              << ",\"iterations\":" << result.iterations
              << ",\"average_microseconds\":" << result.average_microseconds
              << ",\"logical_gib_per_second\":" << result.logical_gib_per_second
              << '}';
  }
  std::cout << "]}\n";
  return 0;
}
