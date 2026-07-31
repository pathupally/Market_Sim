#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#include "marketforge/cuda/smollm2.hpp"

namespace {

[[nodiscard]] bool check_cuda(const cudaError_t status,
                              const std::string_view operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

} // namespace

int main(const int argument_count, const char* const* const arguments) {
  if (argument_count != 2) {
    std::cerr << "usage: marketforge_cuda_smollm2_conformance CHECKPOINT\n";
    return 2;
  }
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

  const auto load_started = std::chrono::steady_clock::now();
  auto model_result = marketforge::cuda::CudaSmolLm2::load(
      std::filesystem::path(arguments[1]), 8, 4);
  if (!model_result) {
    std::cerr << "native CUDA model load failed: "
              << static_cast<std::uint16_t>(model_result.status().code)
              << '\n';
    return 1;
  }
  auto model = std::move(model_result).value();
  const double model_load_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - load_started)
          .count();

  constexpr std::array<std::uint32_t, 4> prompt{0, 1, 2, 3};
  constexpr std::array<std::uint32_t, 3> expected{198, 198, 504};
  std::array<std::uint32_t, 3> generated{};
  const auto inference_started = std::chrono::steady_clock::now();
  auto next = model.prefill(prompt);
  if (!next) {
    return 1;
  }
  generated[0] = next.value();
  for (std::size_t index = 1; index < generated.size(); ++index) {
    next = model.decode(generated[index - 1]);
    if (!next) {
      return 1;
    }
    generated[index] = next.value();
  }
  const double inference_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - inference_started)
          .count();
  if (generated != expected || model.context_length() != 6) {
    std::cerr << "native CUDA greedy token mismatch\n";
    return 1;
  }
  constexpr std::array<std::array<std::uint32_t, 4>, 3> legal_tokens{{
      {{42, 198, 504, 900}},
      {{17, 198, 504, 777}},
      {{3, 198, 504, 1'024}},
  }};
  std::array<std::uint32_t, 3> restricted{};
  if (!model.reset().ok()) {
    return 1;
  }
  next = model.prefill_restricted(prompt, legal_tokens[0]);
  if (!next) {
    return 1;
  }
  restricted[0] = next.value();
  for (std::size_t index = 1; index < restricted.size(); ++index) {
    next = model.decode_restricted(restricted[index - 1],
                                   legal_tokens[index]);
    if (!next) {
      return 1;
    }
    restricted[index] = next.value();
  }
  if (restricted != expected || model.context_length() != 6) {
    std::cerr << "native CUDA restricted output-head token mismatch\n";
    return 1;
  }
  const auto memory = model.memory();
  std::cout << std::fixed << std::setprecision(6)
            << "{\"schema_version\":1,\"result\":\"pass\","
               "\"backend\":\"native_cuda_f16\","
               "\"model\":\"SmolLM2-135M\","
               "\"gpu\":{\"name\":\""
            << properties.name << "\",\"compute_capability\":\""
            << properties.major << '.' << properties.minor
            << "\"},\"prompt_token_ids\":[0,1,2,3],"
               "\"generated_token_ids\":["
            << generated[0] << ',' << generated[1] << ',' << generated[2]
            << "],\"model_load_seconds\":" << model_load_seconds
            << ",\"inference_seconds\":" << inference_seconds
            << ",\"context_length\":" << model.context_length()
            << ",\"memory\":{\"weight_bytes\":" << memory.weight_bytes
            << ",\"kv_bytes\":" << memory.kv_bytes
            << ",\"execution_bytes\":" << memory.execution_bytes
            << ",\"total_device_bytes\":" << memory.total_device_bytes
            << "}}\n";
  return 0;
}
