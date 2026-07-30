#include "marketforge/serving/sequence_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace marketforge::serving {

bool SequenceScheduler::is_terminal(SequencePhase phase) noexcept {
  return phase == SequencePhase::finished ||
         phase == SequencePhase::cancelled;
}

bool SequenceScheduler::is_runnable(const Entry& entry) noexcept {
  return !entry.in_flight && !entry.cancel_pending &&
         (entry.phase == SequencePhase::queued ||
          entry.phase == SequencePhase::decoding);
}

Status SequenceScheduler::enqueue(SequenceRequest request) noexcept {
  if (request.id == 0 || request.prompt_tokens == 0 ||
      request.max_output_tokens == 0 ||
      request.prompt_tokens >
          std::numeric_limits<std::uint32_t>::max() -
              request.max_output_tokens ||
      find(request.id) != nullptr) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  try {
    entries_.push_back(Entry{request});
  } catch (const std::bad_alloc&) {
    return Status::failure(ErrorCode::allocation_failed);
  } catch (const std::length_error&) {
    return Status::failure(ErrorCode::resource_limit);
  }
  return Status::success();
}

Result<InferenceBatch>
SequenceScheduler::schedule(std::size_t maximum_sequences) noexcept {
  if (maximum_sequences == 0) {
    return Result<InferenceBatch>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  if (entries_.empty()) {
    return Result<InferenceBatch>::success(
        InferenceBatch{generation_, {}});
  }

  try {
    const auto limit = std::min(maximum_sequences, entries_.size());
    std::vector<std::size_t> selected;
    selected.reserve(limit);

    const auto start = next_scan_index_ % entries_.size();
    for (std::size_t offset = 0;
         offset < entries_.size() && selected.size() < limit; ++offset) {
      const auto index = (start + offset) % entries_.size();
      if (is_runnable(entries_[index])) {
        selected.push_back(index);
      }
    }

    if (selected.empty()) {
      return Result<InferenceBatch>::success(
          InferenceBatch{generation_, {}});
    }
    if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return Result<InferenceBatch>::failure(
          Status::failure(ErrorCode::arithmetic_overflow));
    }

    InferenceBatch batch;
    batch.generation = generation_ + 1;
    batch.items.reserve(selected.size());
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
      const auto& entry = entries_[selected[slot]];
      const auto work = entry.phase == SequencePhase::queued
                            ? WorkKind::prefill
                            : WorkKind::decode;
      batch.items.push_back(BatchItem{
          entry.request.id,
          work,
          entry.request.prompt_tokens + entry.generated_tokens,
          slot,
      });
    }

    generation_ = batch.generation;
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
      auto& entry = entries_[selected[slot]];
      entry.in_flight = true;
      entry.in_flight_work = batch.items[slot].work;
    }
    next_scan_index_ = (selected.back() + 1) % entries_.size();
    return Result<InferenceBatch>::success(std::move(batch));
  } catch (const std::bad_alloc&) {
    return Result<InferenceBatch>::failure(
        Status::failure(ErrorCode::allocation_failed));
  } catch (const std::length_error&) {
    return Result<InferenceBatch>::failure(
        Status::failure(ErrorCode::resource_limit));
  }
}

Status SequenceScheduler::complete(SequenceId id,
                                   std::uint32_t emitted_tokens,
                                   bool terminal) noexcept {
  auto* entry = find(id);
  if (entry == nullptr || !entry->in_flight ||
      (emitted_tokens == 0 && !terminal)) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  entry->in_flight = false;
  if (entry->cancel_pending) {
    entry->cancel_pending = false;
    entry->phase = SequencePhase::cancelled;
    return Status::success();
  }

  const auto remaining =
      entry->request.max_output_tokens - entry->generated_tokens;
  if (emitted_tokens > remaining) {
    entry->in_flight = true;
    return Status::failure(ErrorCode::invalid_argument);
  }
  entry->generated_tokens += emitted_tokens;
  entry->phase =
      terminal ||
              entry->generated_tokens == entry->request.max_output_tokens
          ? SequencePhase::finished
          : SequencePhase::decoding;
  return Status::success();
}

Status SequenceScheduler::pause(SequenceId id) noexcept {
  auto* entry = find(id);
  if (entry == nullptr || entry->in_flight || is_terminal(entry->phase)) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (entry->phase == SequencePhase::paused) {
    return Status::success();
  }
  entry->resume_phase = entry->phase;
  entry->phase = SequencePhase::paused;
  return Status::success();
}

Status SequenceScheduler::resume(SequenceId id) noexcept {
  auto* entry = find(id);
  if (entry == nullptr || entry->phase != SequencePhase::paused) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  entry->phase = entry->resume_phase;
  return Status::success();
}

Status SequenceScheduler::cancel(SequenceId id) noexcept {
  auto* entry = find(id);
  if (entry == nullptr || entry->phase == SequencePhase::finished) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  if (entry->phase == SequencePhase::cancelled) {
    return Status::success();
  }
  if (entry->in_flight) {
    entry->cancel_pending = true;
    return Status::success();
  }
  entry->phase = SequencePhase::cancelled;
  return Status::success();
}

Result<SequenceState>
SequenceScheduler::state(SequenceId id) const noexcept {
  const auto* entry = find(id);
  if (entry == nullptr) {
    return Result<SequenceState>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  return Result<SequenceState>::success(SequenceState{
      entry->request.id,
      entry->phase,
      entry->request.prompt_tokens,
      entry->generated_tokens,
      entry->request.max_output_tokens,
      entry->in_flight,
      entry->cancel_pending,
  });
}

SchedulerSnapshot SequenceScheduler::snapshot() const noexcept {
  SchedulerSnapshot result;
  result.generation = generation_;
  result.sequence_count = entries_.size();
  for (const auto& entry : entries_) {
    result.in_flight_count += entry.in_flight ? 1U : 0U;
    result.paused_count += entry.phase == SequencePhase::paused ? 1U : 0U;
    result.finished_count +=
        entry.phase == SequencePhase::finished ? 1U : 0U;
    result.cancelled_count +=
        entry.phase == SequencePhase::cancelled ? 1U : 0U;
    result.resident_count += !is_terminal(entry.phase) ? 1U : 0U;
    result.runnable_count += is_runnable(entry) ? 1U : 0U;
  }
  return result;
}

SequenceScheduler::Entry*
SequenceScheduler::find(SequenceId id) noexcept {
  const auto iterator =
      std::find_if(entries_.begin(), entries_.end(),
                   [id](const Entry& entry) { return entry.request.id == id; });
  return iterator == entries_.end() ? nullptr : &*iterator;
}

const SequenceScheduler::Entry*
SequenceScheduler::find(SequenceId id) const noexcept {
  const auto iterator =
      std::find_if(entries_.begin(), entries_.end(),
                   [id](const Entry& entry) { return entry.request.id == id; });
  return iterator == entries_.end() ? nullptr : &*iterator;
}

} // namespace marketforge::serving
