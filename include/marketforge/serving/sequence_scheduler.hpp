#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "marketforge/core/result.hpp"
#include "marketforge/core/status.hpp"

namespace marketforge::serving {

using SequenceId = std::uint64_t;

enum class SequencePhase : std::uint8_t {
  queued,
  decoding,
  paused,
  finished,
  cancelled,
};

enum class WorkKind : std::uint8_t {
  prefill,
  decode,
};

struct SequenceRequest {
  SequenceId id{0};
  std::uint32_t prompt_tokens{0};
  std::uint32_t max_output_tokens{0};

  friend constexpr bool
  operator==(const SequenceRequest&, const SequenceRequest&) = default;
};

struct BatchItem {
  SequenceId id{0};
  WorkKind work{WorkKind::prefill};
  std::uint32_t context_tokens{0};
  std::size_t slot{0};

  friend constexpr bool operator==(const BatchItem&, const BatchItem&) = default;
};

struct InferenceBatch {
  std::uint64_t generation{0};
  std::vector<BatchItem> items;
};

struct SequenceState {
  SequenceId id{0};
  SequencePhase phase{SequencePhase::queued};
  std::uint32_t prompt_tokens{0};
  std::uint32_t generated_tokens{0};
  std::uint32_t max_output_tokens{0};
  bool in_flight{false};
  bool cancel_pending{false};

  friend constexpr bool
  operator==(const SequenceState&, const SequenceState&) = default;
};

struct SchedulerSnapshot {
  std::uint64_t generation{0};
  std::size_t sequence_count{0};
  std::size_t resident_count{0};
  std::size_t runnable_count{0};
  std::size_t in_flight_count{0};
  std::size_t paused_count{0};
  std::size_t finished_count{0};
  std::size_t cancelled_count{0};

  friend constexpr bool
  operator==(const SchedulerSnapshot&, const SchedulerSnapshot&) = default;
};

class SequenceScheduler {
public:
  [[nodiscard]] Status enqueue(SequenceRequest request) noexcept;

  [[nodiscard]] Result<InferenceBatch>
  schedule(std::size_t maximum_sequences) noexcept;

  [[nodiscard]] Status complete(SequenceId id,
                                std::uint32_t emitted_tokens,
                                bool terminal) noexcept;

  [[nodiscard]] Status pause(SequenceId id) noexcept;
  [[nodiscard]] Status resume(SequenceId id) noexcept;
  [[nodiscard]] Status cancel(SequenceId id) noexcept;

  [[nodiscard]] Result<SequenceState> state(SequenceId id) const noexcept;
  [[nodiscard]] SchedulerSnapshot snapshot() const noexcept;

private:
  struct Entry {
    SequenceRequest request;
    SequencePhase phase{SequencePhase::queued};
    SequencePhase resume_phase{SequencePhase::queued};
    WorkKind in_flight_work{WorkKind::prefill};
    std::uint32_t generated_tokens{0};
    bool in_flight{false};
    bool cancel_pending{false};
  };

  [[nodiscard]] static bool is_terminal(SequencePhase phase) noexcept;
  [[nodiscard]] static bool is_runnable(const Entry& entry) noexcept;
  [[nodiscard]] Entry* find(SequenceId id) noexcept;
  [[nodiscard]] const Entry* find(SequenceId id) const noexcept;

  std::vector<Entry> entries_;
  std::size_t next_scan_index_{0};
  std::uint64_t generation_{0};
};

} // namespace marketforge::serving
