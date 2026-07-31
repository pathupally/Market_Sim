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
#include "marketforge/cuda/greedy.hpp"
#include "marketforge/cuda/linear.hpp"

namespace {

using marketforge::cuda::CublasHandle;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

constexpr std::uint64_t hidden_size = 576;
constexpr std::uint64_t vocabulary_size = 49'152;

struct BenchmarkCase {
  std::uint64_t rows;
  std::uint64_t allowed_tokens;
  std::uint32_t iterations;
};

struct Measurement {
  std::uint64_t rows;
  std::uint64_t allowed_tokens_per_row;
  std::uint32_t iterations;
  double full_average_microseconds;
  double restricted_average_microseconds;
  double speedup;
  std::uint64_t full_scores;
  std::uint64_t restricted_scores;
  std::uint64_t materialized_logit_bytes_avoided;
  bool exact_token_parity;
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
  for (std::uint32_t iteration = 0; iteration < 10; ++iteration) {
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

[[nodiscard]] bool run_case(const BenchmarkCase benchmark,
                            const CudaStream& stream,
                            CublasHandle& cublas,
                            const DeviceBuffer& embedding,
                            Measurement& measurement) {
  const auto hidden_elements =
      static_cast<std::size_t>(benchmark.rows * hidden_size);
  const auto logit_elements =
      static_cast<std::size_t>(benchmark.rows * vocabulary_size);
  const auto allowed_elements = static_cast<std::size_t>(
      benchmark.rows * benchmark.allowed_tokens);
  std::vector<__half> hidden(hidden_elements);
  std::vector<std::uint32_t> allowed(allowed_elements);
  std::vector<std::uint32_t> counts(
      static_cast<std::size_t>(benchmark.rows),
      static_cast<std::uint32_t>(benchmark.allowed_tokens));
  std::vector<std::uint32_t> full_tokens(counts.size());
  std::vector<std::uint32_t> restricted_tokens(counts.size());
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    hidden[index] = __float2half_rn(
        0.5F + static_cast<float>(index % 13U) / 32.0F);
  }
  for (std::size_t row = 0; row < counts.size(); ++row) {
    for (std::size_t candidate = 0;
         candidate < benchmark.allowed_tokens; ++candidate) {
      allowed[row * benchmark.allowed_tokens + candidate] =
          static_cast<std::uint32_t>(
              (row * 131U + candidate * 977U + 17U) % vocabulary_size);
    }
  }

  auto hidden_result =
      DeviceBuffer::allocate(hidden_elements * sizeof(__half));
  auto logits_result =
      DeviceBuffer::allocate(logit_elements * sizeof(__half));
  auto allowed_result =
      DeviceBuffer::allocate(allowed_elements * sizeof(std::uint32_t));
  auto counts_result =
      DeviceBuffer::allocate(counts.size() * sizeof(std::uint32_t));
  auto full_result =
      DeviceBuffer::allocate(full_tokens.size() * sizeof(std::uint32_t));
  auto restricted_result =
      DeviceBuffer::allocate(restricted_tokens.size() * sizeof(std::uint32_t));
  if (!hidden_result || !logits_result || !allowed_result || !counts_result ||
      !full_result || !restricted_result) {
    return false;
  }
  DeviceBuffer hidden_device = std::move(hidden_result).value();
  DeviceBuffer logits_device = std::move(logits_result).value();
  DeviceBuffer allowed_device = std::move(allowed_result).value();
  DeviceBuffer counts_device = std::move(counts_result).value();
  DeviceBuffer full_device = std::move(full_result).value();
  DeviceBuffer restricted_device = std::move(restricted_result).value();
  if (!hidden_device
           .copy_from_host_async(hidden.data(),
                                 hidden.size() * sizeof(__half), 0,
                                 stream.handle())
           .ok() ||
      !allowed_device
           .copy_from_host_async(allowed.data(),
                                 allowed.size() * sizeof(std::uint32_t), 0,
                                 stream.handle())
           .ok() ||
      !counts_device
           .copy_from_host_async(counts.data(),
                                 counts.size() * sizeof(std::uint32_t), 0,
                                 stream.handle())
           .ok()) {
    return false;
  }
  const auto full_launch = [&]() {
    return marketforge::cuda::linear_f16(
               hidden_device, embedding, logits_device, benchmark.rows,
               hidden_size, vocabulary_size, cublas, stream.handle())
               .ok() &&
           marketforge::cuda::restricted_greedy_select_f16(
               logits_device, allowed_device, counts_device, full_device,
               benchmark.rows, vocabulary_size, benchmark.allowed_tokens,
               stream.handle())
               .ok();
  };
  const auto restricted_launch = [&]() {
    return marketforge::cuda::restricted_output_head_f16(
               hidden_device, embedding, allowed_device, counts_device,
               restricted_device, benchmark.rows, hidden_size,
               vocabulary_size, benchmark.allowed_tokens, stream.handle())
        .ok();
  };
  if (!full_launch() || !restricted_launch() ||
      !full_device
           .copy_to_host_async(full_tokens.data(),
                               full_tokens.size() * sizeof(std::uint32_t), 0,
                               stream.handle())
           .ok() ||
      !restricted_device
           .copy_to_host_async(
               restricted_tokens.data(),
               restricted_tokens.size() * sizeof(std::uint32_t), 0,
               stream.handle())
           .ok() ||
      !stream.synchronize().ok() || full_tokens != restricted_tokens) {
    return false;
  }

  double full_seconds = 0.0;
  double restricted_seconds = 0.0;
  if (!measure(stream, benchmark.iterations, full_launch, full_seconds) ||
      !measure(stream, benchmark.iterations, restricted_launch,
               restricted_seconds)) {
    return false;
  }
  measurement = Measurement{
      benchmark.rows,
      benchmark.allowed_tokens,
      benchmark.iterations,
      full_seconds * 1.0e6,
      restricted_seconds * 1.0e6,
      full_seconds / restricted_seconds,
      benchmark.rows * vocabulary_size,
      benchmark.rows * benchmark.allowed_tokens,
      benchmark.rows * vocabulary_size * sizeof(__half),
      true,
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
  std::vector<__half> host_embedding(
      static_cast<std::size_t>(vocabulary_size * hidden_size));
  for (std::size_t token = 0; token < vocabulary_size; ++token) {
    const auto token_value = static_cast<float>(token % 101U) / 64.0F;
    for (std::size_t feature = 0; feature < hidden_size; ++feature) {
      host_embedding[token * hidden_size + feature] = __float2half_rn(
          token_value + static_cast<float>(feature % 7U) / 256.0F);
    }
  }
  auto stream_result = CudaStream::create();
  auto cublas_result = CublasHandle::create();
  auto embedding_result =
      DeviceBuffer::allocate(host_embedding.size() * sizeof(__half));
  if (!stream_result || !cublas_result || !embedding_result) {
    return 1;
  }
  CudaStream stream = std::move(stream_result).value();
  CublasHandle cublas = std::move(cublas_result).value();
  DeviceBuffer embedding = std::move(embedding_result).value();
  if (!embedding
           .copy_from_host_async(host_embedding.data(),
                                 host_embedding.size() * sizeof(__half), 0,
                                 stream.handle())
           .ok() ||
      !stream.synchronize().ok()) {
    return 1;
  }

  constexpr std::array cases{
      BenchmarkCase{1, 2, 500},
      BenchmarkCase{1, 8, 500},
      BenchmarkCase{1, 32, 500},
      BenchmarkCase{1, 128, 500},
      BenchmarkCase{16, 2, 300},
      BenchmarkCase{16, 8, 300},
      BenchmarkCase{16, 32, 300},
      BenchmarkCase{16, 128, 300},
      BenchmarkCase{256, 2, 100},
      BenchmarkCase{256, 8, 100},
      BenchmarkCase{256, 32, 100},
      BenchmarkCase{256, 128, 100},
  };
  std::array<Measurement, cases.size()> measurements{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!run_case(cases[index], stream, cublas, embedding,
                  measurements[index])) {
      return 1;
    }
  }

  std::cout << std::fixed << std::setprecision(3)
            << "{\"schema_version\":1,\"result\":\"pass\","
               "\"operator\":\"restricted_output_head_f16\","
               "\"model_shape\":\"SmolLM2-135M\","
               "\"hidden_size\":"
            << hidden_size << ",\"vocabulary_size\":" << vocabulary_size
            << ",\"gpu\":{\"name\":\"" << properties.name
            << "\",\"compute_capability\":\"" << properties.major << '.'
            << properties.minor << "\"},\"measurements\":[";
  for (std::size_t index = 0; index < measurements.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto& result = measurements[index];
    std::cout << "{\"rows\":" << result.rows
              << ",\"allowed_tokens_per_row\":"
              << result.allowed_tokens_per_row
              << ",\"iterations\":" << result.iterations
              << ",\"full_average_microseconds\":"
              << result.full_average_microseconds
              << ",\"restricted_average_microseconds\":"
              << result.restricted_average_microseconds
              << ",\"speedup\":" << result.speedup
              << ",\"full_scores\":" << result.full_scores
              << ",\"restricted_scores\":" << result.restricted_scores
              << ",\"materialized_logit_bytes_avoided\":"
              << result.materialized_logit_bytes_avoided
              << ",\"exact_token_parity\":"
              << (result.exact_token_parity ? "true" : "false") << '}';
  }
  std::cout << "]}\n";
  return 0;
}
