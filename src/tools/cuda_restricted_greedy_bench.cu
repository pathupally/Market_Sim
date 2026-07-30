#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/greedy.hpp"
#include "marketforge/grammar/smollm2_market_actions.hpp"

namespace {

using marketforge::GrammarState;
using marketforge::SmolLm2MarketActionCatalog;
using marketforge::token_id_t;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

struct BenchmarkCase {
  std::uint64_t rows;
  std::uint32_t iterations;
};

struct Measurement {
  std::uint64_t rows;
  std::uint64_t total_allowed_candidates;
  std::uint32_t iterations;
  double average_microseconds;
  double sequences_per_second;
  double candidate_gib_per_second;
};

[[nodiscard]] cudaStream_t
native_stream(const marketforge::cuda::StreamHandle stream) noexcept {
  return reinterpret_cast<cudaStream_t>(stream.value);
}

[[nodiscard]] bool check_cuda(cudaError_t status,
                              std::string_view operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

template <typename Launch>
[[nodiscard]] bool measure(const CudaStream& stream,
                           std::uint32_t iterations, Launch launch,
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

[[nodiscard]] bool collect_states(
    const marketforge::ActionDfa& dfa,
    std::vector<std::vector<token_id_t>>& states,
    std::size_t& maximum_allowed) {
  maximum_allowed = 0;
  states.clear();
  states.reserve(dfa.state_count());
  for (std::uint32_t index = 0; index < dfa.state_count(); ++index) {
    const auto allowed = dfa.allowed(GrammarState{index});
    if (!allowed) {
      return false;
    }
    if (allowed.value().empty()) {
      continue;
    }
    std::vector<token_id_t> tokens;
    tokens.reserve(allowed.value().size());
    for (const auto& arc : allowed.value()) {
      tokens.push_back(arc.token);
    }
    maximum_allowed = std::max(maximum_allowed, tokens.size());
    states.push_back(std::move(tokens));
  }
  return !states.empty() && maximum_allowed != 0;
}

[[nodiscard]] bool run_case(
    const BenchmarkCase benchmark,
    const std::vector<std::vector<token_id_t>>& grammar_states,
    std::size_t maximum_allowed, Measurement& measurement) {
  constexpr auto vocabulary_size =
      SmolLm2MarketActionCatalog::vocabulary_size();
  const auto logits_elements =
      static_cast<std::size_t>(benchmark.rows * vocabulary_size);
  const auto allowed_elements =
      static_cast<std::size_t>(benchmark.rows * maximum_allowed);
  std::vector<__half> logits(logits_elements, __float2half_rn(-4.0F));
  std::vector<std::uint32_t> allowed(
      allowed_elements,
      marketforge::cuda::restricted_greedy_invalid_token_id);
  std::vector<std::uint32_t> counts(
      static_cast<std::size_t>(benchmark.rows));
  std::uint64_t total_allowed = 0;
  for (std::size_t row = 0; row < counts.size(); ++row) {
    const auto state_index =
        row * grammar_states.size() / counts.size();
    const auto& candidates = grammar_states[state_index];
    counts[row] = static_cast<std::uint32_t>(candidates.size());
    total_allowed += candidates.size();
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const auto token = candidates[index];
      allowed[row * maximum_allowed + index] = token;
      logits[row * vocabulary_size + token] = __float2half_rn(
          static_cast<float>((token * 13U + row) % 41U));
    }
  }

  auto stream_result = CudaStream::create();
  auto logits_result =
      DeviceBuffer::allocate(logits.size() * sizeof(__half));
  auto allowed_result =
      DeviceBuffer::allocate(allowed.size() * sizeof(std::uint32_t));
  auto counts_result =
      DeviceBuffer::allocate(counts.size() * sizeof(std::uint32_t));
  auto output_result =
      DeviceBuffer::allocate(counts.size() * sizeof(std::uint32_t));
  if (!stream_result || !logits_result || !allowed_result ||
      !counts_result || !output_result) {
    return false;
  }
  CudaStream stream = std::move(stream_result).value();
  DeviceBuffer logits_device = std::move(logits_result).value();
  DeviceBuffer allowed_device = std::move(allowed_result).value();
  DeviceBuffer counts_device = std::move(counts_result).value();
  DeviceBuffer output_device = std::move(output_result).value();
  if (!logits_device
           .copy_from_host_async(logits.data(),
                                 logits.size() * sizeof(__half), 0,
                                 stream.handle())
           .ok() ||
      !allowed_device
           .copy_from_host_async(
               allowed.data(), allowed.size() * sizeof(std::uint32_t), 0,
               stream.handle())
           .ok() ||
      !counts_device
           .copy_from_host_async(
               counts.data(), counts.size() * sizeof(std::uint32_t), 0,
               stream.handle())
           .ok()) {
    return false;
  }

  double average_seconds = 0.0;
  const auto launch = [&]() {
    return marketforge::cuda::restricted_greedy_select_f16(
               logits_device, allowed_device, counts_device, output_device,
               benchmark.rows, vocabulary_size, maximum_allowed,
               stream.handle())
        .ok();
  };
  if (!measure(stream, benchmark.iterations, launch, average_seconds)) {
    return false;
  }
  const auto logical_bytes =
      static_cast<double>(
          total_allowed * (sizeof(__half) + sizeof(std::uint32_t)) +
          benchmark.rows * (2U * sizeof(std::uint32_t)));
  measurement = Measurement{
      benchmark.rows,
      total_allowed,
      benchmark.iterations,
      average_seconds * 1.0e6,
      static_cast<double>(benchmark.rows) / average_seconds,
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

  auto catalog_result = SmolLm2MarketActionCatalog::create();
  if (!catalog_result) {
    return 1;
  }
  auto catalog = std::move(catalog_result).value();
  std::vector<std::vector<token_id_t>> grammar_states;
  std::size_t maximum_allowed = 0;
  if (!collect_states(catalog.dfa(), grammar_states, maximum_allowed)) {
    return 1;
  }

  constexpr std::array<BenchmarkCase, 3> cases{{
      {1, 2'000},
      {16, 1'000},
      {256, 500},
  }};
  std::array<Measurement, cases.size()> measurements{};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (!run_case(cases[index], grammar_states, maximum_allowed,
                  measurements[index])) {
      return 1;
    }
  }

  std::cout << std::fixed << std::setprecision(3)
            << "{\"schema_version\":1,\"result\":\"pass\","
               "\"operator\":\"restricted_greedy_f16\","
               "\"grammar\":\"smollm2_market_action_v1\","
               "\"vocabulary_size\":"
            << SmolLm2MarketActionCatalog::vocabulary_size()
            << ",\"grammar_states\":" << catalog.dfa().state_count()
            << ",\"grammar_arcs\":" << catalog.dfa().arc_count()
            << ",\"maximum_allowed_tokens\":" << maximum_allowed
            << ",\"gpu\":{\"name\":\"" << properties.name
            << "\",\"compute_capability\":\"" << properties.major << '.'
            << properties.minor << "\"},\"measurements\":[";
  for (std::size_t index = 0; index < measurements.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const auto& result = measurements[index];
    std::cout << "{\"rows\":" << result.rows
              << ",\"total_allowed_candidates\":"
              << result.total_allowed_candidates
              << ",\"iterations\":" << result.iterations
              << ",\"average_microseconds\":"
              << result.average_microseconds
              << ",\"sequences_per_second\":"
              << result.sequences_per_second
              << ",\"candidate_gib_per_second\":"
              << result.candidate_gib_per_second << '}';
  }
  std::cout << "]}\n";
  return 0;
}
