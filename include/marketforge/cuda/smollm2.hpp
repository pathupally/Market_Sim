#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "marketforge/core/result.hpp"
#include "marketforge/cuda/cublas_handle.hpp"
#include "marketforge/cuda/cuda_stream.hpp"
#include "marketforge/cuda/device_buffer.hpp"
#include "marketforge/model/model_spec.hpp"

namespace marketforge::cuda {

struct CudaSmolLm2Memory {
  std::uint64_t weight_bytes{0};
  std::uint64_t kv_bytes{0};
  std::uint64_t execution_bytes{0};
  std::uint64_t total_device_bytes{0};

  friend constexpr bool operator==(const CudaSmolLm2Memory&,
                                   const CudaSmolLm2Memory&) = default;
};

class CudaSmolLm2 {
public:
  // The first prefill call must contain exactly maximum_prefill_tokens.
  [[nodiscard]] static Result<CudaSmolLm2>
  load(const std::filesystem::path& checkpoint,
       std::uint32_t maximum_context,
       std::uint32_t maximum_prefill_tokens);

  CudaSmolLm2(CudaSmolLm2&&) noexcept = default;
  CudaSmolLm2& operator=(CudaSmolLm2&&) noexcept = default;
  CudaSmolLm2(const CudaSmolLm2&) = delete;
  CudaSmolLm2& operator=(const CudaSmolLm2&) = delete;

  [[nodiscard]] Status reset() noexcept;
  [[nodiscard]] Result<std::uint32_t>
  prefill(std::span<const std::uint32_t> token_ids) noexcept;
  [[nodiscard]] Result<std::uint32_t>
  decode(std::uint32_t token_id) noexcept;

  [[nodiscard]] std::uint32_t context_length() const noexcept {
    return context_length_;
  }
  [[nodiscard]] CudaSmolLm2Memory memory() const noexcept {
    return memory_;
  }

private:
  struct LayerStorage {
    DeviceBuffer input_norm;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer output;
    DeviceBuffer post_attention_norm;
    DeviceBuffer gate;
    DeviceBuffer up;
    DeviceBuffer down;
    DeviceBuffer key_cache;
    DeviceBuffer value_cache;
  };

  struct ExecutionBuffers {
    DeviceBuffer token_ids;
    DeviceBuffer positions;
    DeviceBuffer hidden;
    DeviceBuffer normalized;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer attention;
    DeviceBuffer gate;
    DeviceBuffer up;
    std::uint64_t rows{0};
  };

  CudaSmolLm2() noexcept = default;

  [[nodiscard]] static Result<ExecutionBuffers>
  allocate_execution(std::uint64_t rows, const ModelSpec& spec) noexcept;
  [[nodiscard]] Result<std::uint32_t>
  forward(std::span<const std::uint32_t> token_ids,
          ExecutionBuffers& execution) noexcept;

  ModelSpec spec_{};
  CudaStream stream_;
  CublasHandle cublas_;
  DeviceBuffer embedding_;
  DeviceBuffer final_norm_;
  std::vector<LayerStorage> layers_;
  ExecutionBuffers prefill_;
  ExecutionBuffers decode_;
  DeviceBuffer last_hidden_;
  DeviceBuffer last_normalized_;
  DeviceBuffer logits_;
  DeviceBuffer selected_token_;
  std::vector<std::uint32_t> host_positions_;
  std::uint32_t maximum_context_{0};
  std::uint32_t maximum_prefill_tokens_{0};
  std::uint32_t context_length_{0};
  CudaSmolLm2Memory memory_{};
};

} // namespace marketforge::cuda
