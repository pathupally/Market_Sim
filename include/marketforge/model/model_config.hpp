#pragma once

#include <filesystem>
#include <string_view>

#include "marketforge/core/result.hpp"
#include "marketforge/model/model_spec.hpp"

namespace marketforge {

[[nodiscard]] Result<ModelSpec> parse_model_config(std::string_view json_text);

[[nodiscard]] Result<ModelSpec>
load_model_config(const std::filesystem::path& path) noexcept;

} // namespace marketforge
