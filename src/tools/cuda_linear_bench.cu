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

#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/linear.hpp"

namespace {

using marketforge::cuda::CublasHandle;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

constexpr std::uint64_t input_features = 576;
constexpr std::uint64_t output_features = 960;

struct BenchmarkCase {
  std::uint64_t rows;
  std::uint32_t iterations;
};

struct Result {
  std::uint64_t rows;
  std::uint32_t iterations;
  double average_microseconds;
  double tera_flops;
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

[[nodiscard]] bool run_case(const BenchmarkCase benchmark, Result& result) {
  const auto input_elements =
      static_cast<std::size_t>(benchmark.rows * input_features);
  const auto weight_elements =
      static_cast<std::size_t>(output_features * input_features);
  const auto output_elements =
      static_cast<std::size_t>(benchmark.rows * output_features);
  std::vector<__half> input(input_elements);
  std::vector<__half> weight(weight_elements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 17) - 8) /
        32.0F);
  }
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = __float2half_rn(
        static_cast<float>(static_cast<std::int32_t>(index % 13) - 6) /
        64.0F);
  }

  auto stream_result = CudaStream::create();
  auto handle_result = CublasHandle::create();
  auto input_result =
      DeviceBuffer::allocate(input_elements * sizeof(__half));
  auto weight_result =
      DeviceBuffer::allocate(weight_elements * sizeof(__half));
  auto output_result =
      DeviceBuffer::allocate(output_elements * sizeof(__half));
  if (!stream_result || !handle_result || !input_result || !weight_result ||
      !output_result) {
    return false;
  }
  CudaStream stream = std::move(stream_result).value();
  CublasHandle handle = std::move(handle_result).value();
  DeviceBuffer input_device = std::move(input_result).value();
  DeviceBuffer weight_device = std::move(weight_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  if (!input_device
           .copy_from_host_async(input.data(),
                                 input_elements * sizeof(__half), 0,
                                 stream.handle())
           .ok() ||
      !weight_device
           .copy_from_host_async(weight.data(),
                                 weight_elements * sizeof(__half), 0,
                                 stream.handle())
           .ok()) {
    return false;
  }
  for (std::uint32_t iteration = 0; iteration < 20; ++iteration) {
    if (!marketforge::cuda::linear_f16(
             input_device, weight_device, output_device, benchmark.rows,
             input_features, output_features, handle, stream.handle())
             .ok()) {
      return false;
    }
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
  for (std::uint32_t iteration = 0;
       success && iteration < benchmark.iterations; ++iteration) {
    success = marketforge::cuda::linear_f16(
                  input_device, weight_device, output_device, benchmark.rows,
                  input_features, output_features, handle, stream.handle())
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
      static_cast<double>(elapsed_milliseconds) /
      (1'000.0 * static_cast<double>(benchmark.iterations));
  const double floating_point_operations =
      2.0 * static_cast<double>(benchmark.rows) *
      static_cast<double>(input_features) *
      static_cast<double>(output_features);
  result = {
      benchmark.rows,
      benchmark.iterations,
      average_seconds * 1.0e6,
      floating_point_operations / average_seconds / 1.0e12,
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
      {1, 1'000},
      {16, 500},
      {256, 100},
  }};
  std::array<Result, cases.size()> results{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!run_case(cases[index], results[index])) {
      return 1;
    }
  }

  std::cout << std::fixed << std::setprecision(3)
            << "{\"schema_version\":1,\"result\":\"pass\","
               "\"operator\":\"linear_f16_fp32_accumulate\","
               "\"model_shape\":\"SmolLM2-135M fused QKV 576x960\","
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
              << ",\"input_features\":" << input_features
              << ",\"output_features\":" << output_features
              << ",\"iterations\":" << result.iterations
              << ",\"average_microseconds\":" << result.average_microseconds
              << ",\"tera_flops\":" << result.tera_flops << '}';
  }
  std::cout << "]}\n";
  return 0;
}
