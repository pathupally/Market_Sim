#include "test_support.hpp"

#include <filesystem>
#include <type_traits>

#include "marketforge/cpu/smollm2.hpp"

namespace {

using marketforge::CpuSmolLm2;
using marketforge::ErrorCode;

MF_TEST(smollm2_cpu_owner_is_move_only_and_validates_limits_before_io) {
  static_assert(!std::is_copy_constructible_v<CpuSmolLm2>);
  static_assert(!std::is_copy_assignable_v<CpuSmolLm2>);
  static_assert(std::is_nothrow_move_constructible_v<CpuSmolLm2>);
  static_assert(std::is_nothrow_move_assignable_v<CpuSmolLm2>);

  const std::filesystem::path absent =
      "/definitely-not-a-marketforge-checkpoint";
  const auto zero_context = CpuSmolLm2::load(absent, 0, 1);
  MF_CHECK(!zero_context);
  MF_CHECK_EQ(zero_context.status().code, ErrorCode::invalid_argument);

  const auto excessive_prefill = CpuSmolLm2::load(absent, 4, 5);
  MF_CHECK(!excessive_prefill);
  MF_CHECK_EQ(excessive_prefill.status().code, ErrorCode::invalid_argument);

  const auto absent_checkpoint = CpuSmolLm2::load(absent, 4, 4);
  MF_CHECK(!absent_checkpoint);
  MF_CHECK_EQ(absent_checkpoint.status().code, ErrorCode::io_error);
}

} // namespace
