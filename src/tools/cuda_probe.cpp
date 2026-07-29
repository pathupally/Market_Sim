#include <array>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "marketforge/core/status.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/cuda/lifecycle_probe.hpp"

namespace {

using marketforge::Status;
using marketforge::cuda::CudaStream;
using marketforge::cuda::DeviceBuffer;

constexpr std::array<std::uint64_t, 6> required_lengths{0,   1,   255,
                                                        256, 257, 1025};
constexpr std::uint32_t sentinel_before = UINT32_C(0x13579BDF);
constexpr std::uint32_t sentinel_after = UINT32_C(0x2468ACE0);

struct ProbeFailure {
  Status status{};
  std::uint64_t length{0};
  std::uint64_t index{0};
  std::string_view reason{};
};

template <typename T> [[nodiscard]] T&& indirect_move(T& value) noexcept {
  return static_cast<T&&>(value);
}

[[nodiscard]] ProbeFailure run_length(const std::uint64_t length) {
  auto stream_result = CudaStream::create();
  if (!stream_result) {
    return {stream_result.status(), length, 0, "stream_create"};
  }
  auto live_stream_result = CudaStream::create();
  if (!live_stream_result) {
    return {live_stream_result.status(), length, 0, "live_stream_create"};
  }
  CudaStream intermediate = std::move(stream_result).value();
  CudaStream stream = std::move(live_stream_result).value();
  stream = std::move(intermediate);
  stream = indirect_move(stream);
  if (intermediate.handle().valid() || !stream.handle().valid()) {
    return {Status::failure(marketforge::ErrorCode::cuda_backend_failure),
            length, 0, "stream_move_state"};
  }

  if (length == 0) {
    auto zero_result = DeviceBuffer::allocate(0);
    if (!zero_result) {
      return {zero_result.status(), length, 0, "zero_allocate"};
    }
    DeviceBuffer zero = std::move(zero_result).value();
    DeviceBuffer moved = std::move(zero);
    moved = indirect_move(moved);
    if (moved.size_bytes() != 0 || moved.address().valid()) {
      return {Status::failure(marketforge::ErrorCode::cuda_backend_failure),
              length, 0, "zero_allocation_state"};
    }
    const auto zero_copy =
        moved.copy_from_host_async(nullptr, 0, 0, stream.handle());
    if (!zero_copy.ok()) {
      return {zero_copy, length, 0, "zero_copy"};
    }
    return {};
  }

  if (length >
      std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    return {Status::failure(marketforge::ErrorCode::arithmetic_overflow),
            length, 0, "host_size"};
  }

  std::vector<std::uint32_t> input(static_cast<std::size_t>(length));
  for (std::uint64_t index = 0; index < length; ++index) {
    input[static_cast<std::size_t>(index)] =
        static_cast<std::uint32_t>(index) ^ UINT32_C(0xA5A5A5A5);
  }
  std::vector<std::uint32_t> guarded_output(
      static_cast<std::size_t>(length + 2), UINT32_C(0xDEADBEEF));
  guarded_output.front() = sentinel_before;
  guarded_output.back() = sentinel_after;

  const auto input_bytes = length * sizeof(std::uint32_t);
  const auto output_bytes = (length + 2) * sizeof(std::uint32_t);
  auto input_result = DeviceBuffer::allocate(input_bytes);
  if (!input_result) {
    return {input_result.status(), length, 0, "input_allocate"};
  }
  DeviceBuffer input_source = std::move(input_result).value();
  DeviceBuffer input_device = std::move(input_source);
  input_device = indirect_move(input_device);

  auto output_result = DeviceBuffer::allocate(output_bytes);
  if (!output_result) {
    return {output_result.status(), length, 0, "output_allocate"};
  }
  auto live_result = DeviceBuffer::allocate(sizeof(std::uint32_t));
  if (!live_result) {
    return {live_result.status(), length, 0, "live_allocate"};
  }
  DeviceBuffer output_source = std::move(output_result).value();
  DeviceBuffer output_device = std::move(live_result).value();
  output_device = std::move(output_source);

  auto status = input_device.copy_from_host_async(input.data(), input_bytes, 0,
                                                  stream.handle());
  if (!status.ok()) {
    return {status, length, 0, "upload_input"};
  }
  status = output_device.copy_from_host_async(guarded_output.data(),
                                              output_bytes, 0, stream.handle());
  if (!status.ok()) {
    return {status, length, 0, "upload_sentinels"};
  }
  status = marketforge::cuda::launch_lifecycle_probe(
      input_device, output_device, length, stream.handle());
  if (!status.ok()) {
    return {status, length, 0, "launch"};
  }
  status = output_device.copy_to_host_async(guarded_output.data(), output_bytes,
                                            0, stream.handle());
  if (!status.ok()) {
    return {status, length, 0, "download"};
  }
  status = stream.synchronize();
  if (!status.ok()) {
    return {status, length, 0, "synchronize"};
  }

  if (guarded_output.front() != sentinel_before) {
    return {Status::failure(marketforge::ErrorCode::cuda_backend_failure),
            length, 0, "sentinel_before"};
  }
  for (std::uint64_t index = 0; index < length; ++index) {
    const auto expected =
        input[static_cast<std::size_t>(index)] * UINT32_C(3) + UINT32_C(7);
    if (guarded_output[static_cast<std::size_t>(index + 1)] != expected) {
      return {Status::failure(marketforge::ErrorCode::cuda_backend_failure),
              length, index, "known_answer"};
    }
  }
  if (guarded_output.back() != sentinel_after) {
    return {Status::failure(marketforge::ErrorCode::cuda_backend_failure),
            length, length + 1, "sentinel_after"};
  }
  return {};
}

[[nodiscard]] bool parse_repetitions(const std::span<char*> arguments,
                                     std::uint64_t& repetitions) {
  if (arguments.size() == 1) {
    return true;
  }
  if (arguments.size() != 3 ||
      std::string_view(arguments[1]) != "--repetitions") {
    return false;
  }
  const std::string_view value(arguments[2]);
  const auto conversion =
      std::from_chars(value.data(), value.data() + value.size(), repetitions);
  return conversion.ec == std::errc{} &&
         conversion.ptr == value.data() + value.size() && repetitions > 0;
}

} // namespace

int main(const int argc, char** argv) {
  std::uint64_t repetitions = 1;
  if (!parse_repetitions(std::span<char*>(argv, static_cast<std::size_t>(argc)),
                         repetitions)) {
    std::cerr << "{\"schema_version\":1,\"result\":\"fail\","
                 "\"reason\":\"usage\"}\n";
    return 2;
  }

  for (std::uint64_t repetition = 0; repetition < repetitions; ++repetition) {
    for (const auto length : required_lengths) {
      const auto failure = run_length(length);
      if (!failure.status.ok()) {
        std::cout << "{\"schema_version\":1,\"result\":\"fail\","
                     "\"length\":"
                  << failure.length << ",\"repetition\":" << repetition
                  << ",\"index\":" << failure.index << ",\"reason\":\""
                  << failure.reason << "\",\"error_code\":"
                  << static_cast<std::uint16_t>(failure.status.code)
                  << ",\"detail\":" << failure.status.detail << "}\n";
        return 1;
      }
    }
  }

  std::cout << "{\"schema_version\":1,\"result\":\"pass\","
               "\"known_answer_lengths\":[0,1,255,256,257,1025],"
               "\"sentinels\":\"pass\",\"lifecycle_repetitions\":"
            << repetitions << "}\n";
  return 0;
}
