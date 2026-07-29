#include <cstddef>
#include <cstdint>
#include <span>

#include "marketforge/model/safetensors.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      const std::size_t size) {
  const auto bytes = std::as_bytes(std::span(data, size));
  static_cast<void>(
      marketforge::parse_safetensors_metadata(bytes, 1024ULL * 1024ULL));
  return 0;
}
