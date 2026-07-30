#include "test_support.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "marketforge/core/status.hpp"
#include "marketforge/serving/sequence_scheduler.hpp"

namespace {

using marketforge::ErrorCode;
using marketforge::serving::BatchItem;
using marketforge::serving::SequencePhase;
using marketforge::serving::SequenceRequest;
using marketforge::serving::SequenceScheduler;
using marketforge::serving::WorkKind;

void enqueue(SequenceScheduler& scheduler, std::uint64_t id,
             std::uint32_t prompt_tokens = 4,
             std::uint32_t maximum_output_tokens = 8) {
  MF_CHECK(scheduler
               .enqueue(SequenceRequest{id, prompt_tokens,
                                        maximum_output_tokens})
               .ok());
}

std::vector<BatchItem> schedule(SequenceScheduler& scheduler,
                                std::size_t capacity) {
  auto result = scheduler.schedule(capacity);
  MF_CHECK(result);
  return std::move(result).value().items;
}

MF_TEST(sequence_scheduler_validates_requests_and_capacity) {
  SequenceScheduler scheduler;
  MF_CHECK_EQ(scheduler.enqueue(SequenceRequest{}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(scheduler.enqueue(SequenceRequest{1, 0, 1}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(scheduler.enqueue(SequenceRequest{1, 1, 0}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(
      scheduler
          .enqueue(SequenceRequest{1,
                                   std::numeric_limits<std::uint32_t>::max(),
                                   1})
          .code,
      ErrorCode::invalid_argument);
  enqueue(scheduler, 1);
  MF_CHECK_EQ(scheduler.enqueue(SequenceRequest{1, 2, 2}).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(scheduler.schedule(0).status().code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(scheduler.state(99).status().code,
              ErrorCode::invalid_argument);
}

MF_TEST(sequence_scheduler_is_deterministic_and_round_robin_fair) {
  SequenceScheduler scheduler;
  enqueue(scheduler, 10);
  enqueue(scheduler, 20);
  enqueue(scheduler, 30);

  const auto first = schedule(scheduler, 2);
  MF_CHECK_EQ(first.size(), 2U);
  MF_CHECK_EQ(first[0], (BatchItem{10, WorkKind::prefill, 4, 0}));
  MF_CHECK_EQ(first[1], (BatchItem{20, WorkKind::prefill, 4, 1}));
  MF_CHECK(scheduler.complete(10, 1, false).ok());
  MF_CHECK(scheduler.complete(20, 1, false).ok());

  const auto second = schedule(scheduler, 2);
  MF_CHECK_EQ(second.size(), 2U);
  MF_CHECK_EQ(second[0], (BatchItem{30, WorkKind::prefill, 4, 0}));
  MF_CHECK_EQ(second[1], (BatchItem{10, WorkKind::decode, 5, 1}));
  MF_CHECK(scheduler.complete(30, 1, false).ok());
  MF_CHECK(scheduler.complete(10, 1, false).ok());

  const auto third = schedule(scheduler, 2);
  MF_CHECK_EQ(third.size(), 2U);
  MF_CHECK_EQ(third[0], (BatchItem{20, WorkKind::decode, 5, 0}));
  MF_CHECK_EQ(third[1], (BatchItem{30, WorkKind::decode, 5, 1}));
}

MF_TEST(sequence_scheduler_never_dispatches_in_flight_or_paused_work) {
  SequenceScheduler scheduler;
  enqueue(scheduler, 1);
  enqueue(scheduler, 2);

  const auto first = schedule(scheduler, 2);
  MF_CHECK_EQ(first.size(), 2U);
  MF_CHECK(schedule(scheduler, 2).empty());
  MF_CHECK_EQ(scheduler.pause(1).code, ErrorCode::invalid_argument);
  MF_CHECK(scheduler.complete(1, 1, false).ok());
  MF_CHECK(scheduler.complete(2, 1, false).ok());
  MF_CHECK(scheduler.pause(1).ok());

  const auto second = schedule(scheduler, 2);
  MF_CHECK_EQ(second.size(), 1U);
  MF_CHECK_EQ(second[0].id, 2U);
  MF_CHECK(scheduler.complete(2, 1, false).ok());
  MF_CHECK(scheduler.resume(1).ok());
  MF_CHECK_EQ(schedule(scheduler, 1)[0].id, 1U);
}

MF_TEST(sequence_scheduler_commits_completion_or_cancellation_atomically) {
  SequenceScheduler scheduler;
  enqueue(scheduler, 1, 5, 2);
  enqueue(scheduler, 2, 7, 2);

  const auto first = schedule(scheduler, 2);
  MF_CHECK_EQ(first.size(), 2U);
  MF_CHECK(scheduler.cancel(1).ok());
  MF_CHECK(scheduler.state(1).value().cancel_pending);
  MF_CHECK(scheduler.complete(1, 1, false).ok());
  MF_CHECK_EQ(scheduler.state(1).value().phase, SequencePhase::cancelled);
  MF_CHECK_EQ(scheduler.state(1).value().generated_tokens, 0U);

  MF_CHECK_EQ(scheduler.complete(2, 3, false).code,
              ErrorCode::invalid_argument);
  MF_CHECK(scheduler.state(2).value().in_flight);
  MF_CHECK_EQ(scheduler.state(2).value().generated_tokens, 0U);
  MF_CHECK(scheduler.complete(2, 1, false).ok());
  MF_CHECK_EQ(scheduler.state(2).value().phase, SequencePhase::decoding);
  MF_CHECK_EQ(scheduler.state(2).value().generated_tokens, 1U);

  MF_CHECK_EQ(schedule(scheduler, 1)[0].id, 2U);
  MF_CHECK(scheduler.complete(2, 1, false).ok());
  MF_CHECK_EQ(scheduler.state(2).value().phase, SequencePhase::finished);
}

MF_TEST(sequence_scheduler_tracks_lifecycle_snapshot_exactly) {
  SequenceScheduler scheduler;
  enqueue(scheduler, 1);
  enqueue(scheduler, 2);
  enqueue(scheduler, 3);
  MF_CHECK(scheduler.pause(3).ok());

  const auto first = scheduler.schedule(2);
  MF_CHECK(first);
  MF_CHECK_EQ(first.value().generation, 1U);
  MF_CHECK(scheduler.cancel(2).ok());
  auto snapshot = scheduler.snapshot();
  MF_CHECK_EQ(snapshot.sequence_count, 3U);
  MF_CHECK_EQ(snapshot.resident_count, 3U);
  MF_CHECK_EQ(snapshot.runnable_count, 0U);
  MF_CHECK_EQ(snapshot.in_flight_count, 2U);
  MF_CHECK_EQ(snapshot.paused_count, 1U);

  MF_CHECK(scheduler.complete(1, 0, true).ok());
  MF_CHECK(scheduler.complete(2, 1, false).ok());
  snapshot = scheduler.snapshot();
  MF_CHECK_EQ(snapshot.generation, 1U);
  MF_CHECK_EQ(snapshot.resident_count, 1U);
  MF_CHECK_EQ(snapshot.finished_count, 1U);
  MF_CHECK_EQ(snapshot.cancelled_count, 1U);
  MF_CHECK_EQ(snapshot.paused_count, 1U);
}

MF_TEST(sequence_scheduler_bounded_batches_eventually_visit_every_sequence) {
  SequenceScheduler scheduler;
  constexpr std::size_t sequence_count = 127;
  std::array<bool, sequence_count> seen{};
  for (std::size_t index = 0; index < sequence_count; ++index) {
    enqueue(scheduler, index + 1);
  }

  for (std::size_t round = 0; round < 13; ++round) {
    const auto batch = schedule(scheduler, 10);
    MF_CHECK_EQ(batch.size(), 10U);
    for (const auto& item : batch) {
      seen[item.id - 1] = true;
      MF_CHECK(scheduler.complete(item.id, 1, false).ok());
    }
  }
  for (const auto was_seen : seen) {
    MF_CHECK(was_seen);
  }
}

} // namespace
