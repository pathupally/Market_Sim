#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MARKETFORGE_ADDRESS_SANITIZER_ACTIVE 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define MARKETFORGE_ADDRESS_SANITIZER_ACTIVE 1
#endif
#if !defined(MARKETFORGE_ADDRESS_SANITIZER_ACTIVE)
#define MARKETFORGE_ADDRESS_SANITIZER_ACTIVE 0
#endif

#if MARKETFORGE_ADDRESS_SANITIZER_ACTIVE
#include <sanitizer/allocator_interface.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <malloc.h>
#include <unistd.h>
#endif

#include "marketforge/core/mapped_file.hpp"
#include "marketforge/cpu/smollm2.hpp"
#include "marketforge/model/safetensors.hpp"

namespace {

using marketforge::ConstTensorView;
using marketforge::CpuSmolLm2;
using marketforge::DType;
using marketforge::ErrorCode;
using marketforge::GreedyToken;
using marketforge::SafeTensorFile;

struct ComparisonStats {
  float maximum_absolute_error{0.0F};
  float maximum_scaled_error{0.0F};
};

std::uint64_t current_rss_bytes() noexcept {
#if defined(__APPLE__)
  mach_task_basic_info_data_t information{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&information),
                &count) != KERN_SUCCESS) {
    return 0;
  }
  return information.resident_size;
#elif defined(__linux__)
  std::FILE* const status = std::fopen("/proc/self/statm", "r");
  if (status == nullptr) {
    return 0;
  }
  unsigned long total_pages = 0;
  unsigned long resident_pages = 0;
  const int parsed =
      std::fscanf(status, "%lu %lu", &total_pages, &resident_pages);
  static_cast<void>(total_pages);
  std::fclose(status);
  if (parsed != 2) {
    return 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 ||
      resident_pages > std::numeric_limits<std::uint64_t>::max() /
                           static_cast<std::uint64_t>(page_size)) {
    return 0;
  }
  return static_cast<std::uint64_t>(resident_pages) *
         static_cast<std::uint64_t>(page_size);
#else
  return 0;
#endif
}

std::uint64_t current_heap_bytes() noexcept {
#if MARKETFORGE_ADDRESS_SANITIZER_ACTIVE
  return __sanitizer_get_current_allocated_bytes();
#elif defined(__APPLE__)
  vm_address_t* zone_addresses = nullptr;
  unsigned zone_count = 0;
  if (malloc_get_all_zones(mach_task_self(), nullptr, &zone_addresses,
                           &zone_count) != KERN_SUCCESS) {
    return 0;
  }
  std::uint64_t total = 0;
  for (unsigned index = 0; index < zone_count; ++index) {
    malloc_statistics_t statistics{};
    auto* const zone = reinterpret_cast<malloc_zone_t*>(zone_addresses[index]);
    malloc_zone_statistics(zone, &statistics);
    if (statistics.size_in_use >
        std::numeric_limits<std::uint64_t>::max() - total) {
      return 0;
    }
    total += statistics.size_in_use;
  }
  return total;
#elif defined(__linux__)
  const auto statistics = mallinfo2();
  return statistics.uordblks;
#else
  return 0;
#endif
}

marketforge::Result<SafeTensorFile>
open_fixture(const std::filesystem::path& path) {
  auto mapping = marketforge::MappedFile::open_read_only(path);
  if (!mapping) {
    return marketforge::Result<SafeTensorFile>::failure(mapping.status());
  }
  return SafeTensorFile::parse(std::move(mapping).value());
}

ConstTensorView require_tensor(const SafeTensorFile& fixture,
                               const std::string_view name, const DType dtype,
                               const std::uint8_t rank) {
  const auto tensor = fixture.tensor(name);
  if (!tensor || tensor.value().dtype != dtype ||
      tensor.value().shape.rank != rank) {
    std::cerr << "invalid fixture tensor: " << name << '\n';
    std::exit(EXIT_FAILURE);
  }
  return tensor.value();
}

std::vector<std::uint32_t> copy_nonnegative_i32(const ConstTensorView view) {
  const auto count = marketforge::checked_numel(view.shape);
  if (!count) {
    std::cerr << "invalid integer fixture shape\n";
    std::exit(EXIT_FAILURE);
  }
  const auto* const bytes = static_cast<const std::byte*>(view.data);
  std::vector<std::uint32_t> values(static_cast<std::size_t>(count.value()));
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes + index * sizeof(value), sizeof(value));
    if (value < 0) {
      std::cerr << "negative fixture token\n";
      std::exit(EXIT_FAILURE);
    }
    values[index] = static_cast<std::uint32_t>(value);
  }
  return values;
}

bool compare_logits(const std::span<const float> actual,
                    const std::span<const float> expected,
                    const float absolute_tolerance,
                    const float relative_tolerance,
                    ComparisonStats& stats) noexcept {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      std::cerr << "non-finite logit at " << index << '\n';
      return false;
    }
    const float absolute = std::abs(actual[index] - expected[index]);
    const float allowed =
        absolute_tolerance + relative_tolerance * std::abs(expected[index]);
    stats.maximum_absolute_error =
        std::max(stats.maximum_absolute_error, absolute);
    stats.maximum_scaled_error =
        std::max(stats.maximum_scaled_error, absolute / allowed);
    if (absolute > allowed) {
      std::cerr << "logit mismatch at " << index << ": actual " << actual[index]
                << ", expected " << expected[index] << ", absolute error "
                << absolute << ", allowed " << allowed << '\n';
      return false;
    }
  }
  return true;
}

bool check_step(const GreedyToken actual, const std::uint32_t expected_token,
                const float expected_margin,
                const std::span<const float> actual_logits,
                const std::span<const float> expected_logits,
                const float absolute_tolerance, const float relative_tolerance,
                ComparisonStats& stats) noexcept {
  if (actual.token_id != expected_token) {
    std::cerr << "greedy token mismatch: actual " << actual.token_id
              << ", expected " << expected_token << '\n';
    return false;
  }
  const float actual_margin = actual.logit - actual.runner_up_logit;
  if (actual_margin < 0.5F || expected_margin < 0.5F) {
    std::cerr << "fixture or runtime margin is not qualified\n";
    return false;
  }
  return compare_logits(actual_logits, expected_logits, absolute_tolerance,
                        relative_tolerance, stats);
}

bool run_and_compare(CpuSmolLm2& model,
                     const std::span<const std::uint32_t> prompt,
                     const std::span<const std::uint32_t> expected_tokens,
                     const std::span<const float> expected_logits,
                     const std::span<const float> expected_margins,
                     const float absolute_tolerance,
                     const float relative_tolerance,
                     ComparisonStats& stats) noexcept {
  if (!model.reset().ok()) {
    return false;
  }
  auto result = model.prefill(prompt);
  for (std::size_t step = 0; step < expected_tokens.size(); ++step) {
    if (!result) {
      std::cerr << "model forward failed at step " << step << ", error "
                << static_cast<unsigned>(result.status().code) << '\n';
      return false;
    }
    const auto expected_row = expected_logits.subspan(
        step * model.logits().size(), model.logits().size());
    if (!check_step(result.value(), expected_tokens[step],
                    expected_margins[step], model.logits(), expected_row,
                    absolute_tolerance, relative_tolerance, stats)) {
      return false;
    }
    if (step + 1 < expected_tokens.size()) {
      result = model.decode(result.value().token_id);
    }
  }
  return true;
}

bool run_tokens_only(CpuSmolLm2& model,
                     const std::span<const std::uint32_t> prompt,
                     const std::span<const std::uint32_t> expected) noexcept {
  if (!model.reset().ok()) {
    return false;
  }
  auto result = model.prefill(prompt);
  for (std::size_t step = 0; step < expected.size(); ++step) {
    if (!result || result.value().token_id != expected[step]) {
      return false;
    }
    if (step + 1 < expected.size()) {
      result = model.decode(result.value().token_id);
    }
  }
  return true;
}

template <typename Operation>
bool rejected_call_preserves_observable_state(CpuSmolLm2& model,
                                              const std::string_view label,
                                              Operation&& operation) {
  const auto context_before = model.context_length();
  const auto memory_before = model.memory();
  const auto* const logits_pointer_before = model.logits().data();
  const std::vector<float> logits_before(model.logits().begin(),
                                         model.logits().end());

  const auto rejected = operation();
  if (rejected || rejected.status().code != ErrorCode::invalid_argument) {
    std::cerr << label << " was not rejected as invalid_argument\n";
    return false;
  }
  if (model.context_length() != context_before ||
      model.memory() != memory_before ||
      model.logits().data() != logits_pointer_before ||
      !std::equal(model.logits().begin(), model.logits().end(),
                  logits_before.begin(), logits_before.end())) {
    std::cerr << label << " mutated observable model state\n";
    return false;
  }
  return true;
}

bool check_lifecycle_faults(CpuSmolLm2& model,
                            const std::span<const std::uint32_t> prompt) {
  if (!model.reset().ok()) {
    return false;
  }
  if (!rejected_call_preserves_observable_state(
          model, "decode before prefill", [&]() { return model.decode(0); })) {
    return false;
  }

  const std::array<std::uint32_t, 1> invalid_prompt{
      marketforge::smollm2_135m_profile().spec.vocabulary_size};
  if (!rejected_call_preserves_observable_state(
          model, "invalid prefill token",
          [&]() { return model.prefill(invalid_prompt); })) {
    return false;
  }

  const auto prefill = model.prefill(prompt);
  if (!prefill) {
    std::cerr << "lifecycle prefill failed\n";
    return false;
  }
  if (!rejected_call_preserves_observable_state(
          model, "second prefill", [&]() { return model.prefill(prompt); }) ||
      !rejected_call_preserves_observable_state(
          model, "invalid decode token", [&]() {
            return model.decode(
                marketforge::smollm2_135m_profile().spec.vocabulary_size);
          })) {
    return false;
  }

  while (model.context_length() < model.maximum_context()) {
    if (!model.decode(0)) {
      std::cerr << "failed to fill context for overflow check\n";
      return false;
    }
  }
  if (!rejected_call_preserves_observable_state(
          model, "context overflow", [&]() { return model.decode(0); })) {
    return false;
  }
  if (!model.reset().ok() || model.context_length() != 0) {
    std::cerr << "reset after lifecycle checks failed\n";
    return false;
  }
  return true;
}

} // namespace

int main(const int argument_count, const char* const* arguments) {
  if (argument_count < 3 || argument_count > 4) {
    std::cerr << "usage: marketforge_smollm2_conformance "
                 "CHECKPOINT FIXTURE [REPEATS]\n";
    return EXIT_FAILURE;
  }
  const std::uint32_t repeats =
      argument_count == 4
          ? static_cast<std::uint32_t>(std::strtoul(arguments[3], nullptr, 10))
          : 20;
  if (repeats == 0 || repeats > 1'000) {
    std::cerr << "repeat count must be in [1, 1000]\n";
    return EXIT_FAILURE;
  }

  auto fixture_result = open_fixture(arguments[2]);
  if (!fixture_result) {
    std::cerr << "fixture rejected; error "
              << static_cast<unsigned>(fixture_result.status().code) << '\n';
    return EXIT_FAILURE;
  }
  auto fixture = std::move(fixture_result).value();
  const auto prompt_view =
      require_tensor(fixture, "input.token_ids", DType::i32, 1);
  const auto tokens_view =
      require_tensor(fixture, "expected.tokens", DType::i32, 1);
  const auto logits_view =
      require_tensor(fixture, "expected.logits", DType::f32, 2);
  const auto margins_view =
      require_tensor(fixture, "expected.margins", DType::f32, 1);
  const auto absolute_view =
      require_tensor(fixture, "tolerance.logits_absolute", DType::f32, 1);
  const auto relative_view =
      require_tensor(fixture, "tolerance.logits_relative", DType::f32, 1);

  auto prompt = copy_nonnegative_i32(prompt_view);
  auto expected_tokens = copy_nonnegative_i32(tokens_view);
  const auto* const expected_logits =
      static_cast<const float*>(logits_view.data);
  const auto* const expected_margins =
      static_cast<const float*>(margins_view.data);
  const float absolute_tolerance =
      *static_cast<const float*>(absolute_view.data);
  const float relative_tolerance =
      *static_cast<const float*>(relative_view.data);
  if (prompt.empty() || expected_tokens.empty() ||
      logits_view.shape.extents[0] != expected_tokens.size() ||
      logits_view.shape.extents[1] !=
          marketforge::smollm2_135m_profile().spec.vocabulary_size ||
      margins_view.shape.extents[0] != expected_tokens.size() ||
      absolute_view.shape.extents[0] != 1 ||
      relative_view.shape.extents[0] != 1) {
    std::cerr << "fixture shape contract failed\n";
    return EXIT_FAILURE;
  }

  const std::uint32_t maximum_context =
      static_cast<std::uint32_t>(prompt.size() + expected_tokens.size());
  auto model_result =
      CpuSmolLm2::load(std::filesystem::path(arguments[1]), maximum_context,
                       static_cast<std::uint32_t>(prompt.size()));
  if (!model_result) {
    std::cerr << "model load failed; error "
              << static_cast<unsigned>(model_result.status().code)
              << ", detail " << model_result.status().detail << '\n';
    return EXIT_FAILURE;
  }
  auto model = std::move(model_result).value();
  if (!check_lifecycle_faults(model, prompt)) {
    return EXIT_FAILURE;
  }
  const auto expected_logit_span = std::span<const float>(
      expected_logits,
      expected_tokens.size() *
          marketforge::smollm2_135m_profile().spec.vocabulary_size);
  const auto expected_margin_span =
      std::span<const float>(expected_margins, expected_tokens.size());

  ComparisonStats comparison;
  if (!run_and_compare(model, prompt, expected_tokens, expected_logit_span,
                       expected_margin_span, absolute_tolerance,
                       relative_tolerance, comparison)) {
    return EXIT_FAILURE;
  }

  constexpr std::uint32_t warmup_cycles = 10;
  for (std::uint32_t warmup = 0; warmup < warmup_cycles; ++warmup) {
    if (!run_tokens_only(model, prompt, expected_tokens)) {
      std::cerr << "warmup token parity failed\n";
      return EXIT_FAILURE;
    }
  }
  const auto memory_before = model.memory();
  const float* const logits_pointer = model.logits().data();
  const std::uint64_t heap_before = current_heap_bytes();
  const std::uint64_t rss_before = current_rss_bytes();
  const auto run_repeat_window = [&]() {
    for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
      if (!run_tokens_only(model, prompt, expected_tokens) ||
          model.memory() != memory_before ||
          model.logits().data() != logits_pointer) {
        std::cerr << "repeated decode storage/token stability failed at repeat "
                  << repeat << '\n';
        return false;
      }
    }
    return true;
  };
  if (!run_repeat_window()) {
    return EXIT_FAILURE;
  }
  const std::uint64_t heap_middle = current_heap_bytes();
  const std::uint64_t rss_middle = current_rss_bytes();
  if (!run_repeat_window()) {
    return EXIT_FAILURE;
  }
  const std::uint64_t heap_after = current_heap_bytes();
  const std::uint64_t rss_after = current_rss_bytes();
  const std::uint64_t heap_plateau_growth =
      heap_after > heap_middle ? heap_after - heap_middle : 0;
  const std::uint64_t rss_first_window_growth =
      rss_middle > rss_before ? rss_middle - rss_before : 0;
  const std::uint64_t rss_plateau_growth =
      rss_after > rss_middle ? rss_after - rss_middle : 0;
  constexpr std::uint64_t maximum_heap_growth = 1ULL * 1024ULL * 1024ULL;
  if (MARKETFORGE_ADDRESS_SANITIZER_ACTIVE != 0 && heap_middle != 0 &&
      heap_after != 0 && heap_plateau_growth > maximum_heap_growth) {
    std::cerr << "live heap did not plateau: second repeat window grew by "
              << heap_plateau_growth << " bytes; ceiling is "
              << maximum_heap_growth << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "SmolLM2 PR 4 conformance passed\n"
            << "  prompt tokens: " << prompt.size() << '\n'
            << "  greedy steps: " << expected_tokens.size() << '\n'
            << "  lifecycle rejection checks: 5\n"
            << "  compared logits: " << expected_logit_span.size() << '\n'
            << "  maximum absolute logit error: "
            << comparison.maximum_absolute_error << '\n'
            << "  maximum tolerance-scaled error: "
            << comparison.maximum_scaled_error << '\n'
            << "  warmup decode cycles: " << warmup_cycles << '\n'
            << "  measured decode cycles: " << repeats * 2 << '\n'
            << "  fixed owned bytes: " << memory_before.total_owned_bytes
            << '\n'
            << "  live heap before repeats: " << heap_before << '\n'
            << "  live heap between repeat windows: " << heap_middle << '\n'
            << "  live heap after repeats: " << heap_after << '\n'
            << "  plateau-window live-heap growth: " << heap_plateau_growth
            << (MARKETFORGE_ADDRESS_SANITIZER_ACTIVE != 0
                    ? " (ASan allocation gate)\n"
                    : " (system-library-inclusive, observational)\n")
            << "  RSS before repeats: " << rss_before << '\n'
            << "  RSS between repeat windows: " << rss_middle << '\n'
            << "  RSS after repeats: " << rss_after << '\n'
            << "  first-window RSS growth: " << rss_first_window_growth << '\n'
            << "  plateau-window RSS growth: " << rss_plateau_growth
            << " (observational)\n";
  return EXIT_SUCCESS;
}
