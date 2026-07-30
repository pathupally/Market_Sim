#include "marketforge/serving/paged_kv_cache.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace marketforge::serving {

namespace {

[[nodiscard]] bool checked_add(std::uint64_t left, std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] std::uint64_t
pages_for(std::uint32_t tokens, std::uint32_t tokens_per_page) noexcept {
  return (static_cast<std::uint64_t>(tokens) + tokens_per_page - 1U) /
         tokens_per_page;
}

} // namespace

Result<PagedKvCache>
PagedKvCache::create(std::uint32_t page_count,
                     std::uint32_t tokens_per_page) noexcept {
  if (page_count == 0 || tokens_per_page == 0) {
    return Result<PagedKvCache>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  try {
    return Result<PagedKvCache>::success(PagedKvCache(
        tokens_per_page, std::vector<Page>(page_count)));
  } catch (const std::bad_alloc&) {
    return Result<PagedKvCache>::failure(
        Status::failure(ErrorCode::allocation_failed));
  } catch (const std::length_error&) {
    return Result<PagedKvCache>::failure(
        Status::failure(ErrorCode::resource_limit));
  }
}

Status PagedKvCache::create_sequence(SequenceId sequence_id,
                                     PrefixId prefix_id) noexcept {
  if (sequence_id == 0 || find_sequence(sequence_id) != nullptr) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  auto* prefix = prefix_id == 0 ? nullptr : find_prefix(prefix_id);
  if (prefix_id != 0 && (prefix == nullptr ||
                         prefix->attachments ==
                             std::numeric_limits<std::uint64_t>::max())) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  try {
    sequences_.push_back(Sequence{
        sequence_id,
        prefix_id,
        prefix == nullptr ? 0U : prefix->token_count,
        prefix == nullptr ? 0U : prefix->token_count,
        {},
    });
  } catch (const std::bad_alloc&) {
    return Status::failure(ErrorCode::allocation_failed);
  } catch (const std::length_error&) {
    return Status::failure(ErrorCode::resource_limit);
  }
  if (prefix != nullptr) {
    ++prefix->attachments;
  }
  return Status::success();
}

Status PagedKvCache::destroy_sequence(SequenceId sequence_id) noexcept {
  const auto iterator =
      std::find_if(sequences_.begin(), sequences_.end(),
                   [sequence_id](const Sequence& sequence) {
                     return sequence.id == sequence_id;
                   });
  if (iterator == sequences_.end() ||
      find_sequence_reservation(sequence_id) != nullptr) {
    return Status::failure(ErrorCode::invalid_argument);
  }

  Prefix* prefix = nullptr;
  if (iterator->prefix_id != 0) {
    prefix = find_prefix(iterator->prefix_id);
    if (prefix == nullptr || prefix->attachments == 0) {
      return Status::failure(ErrorCode::invalid_format);
    }
  }
  for (const auto page_id : iterator->suffix_pages) {
    if (page_id >= pages_.size() ||
        pages_[page_id] !=
            Page{PageState::mutable_sequence, sequence_id}) {
      return Status::failure(ErrorCode::invalid_format);
    }
  }
  for (const auto page_id : iterator->suffix_pages) {
    pages_[page_id] = Page{};
  }
  if (prefix != nullptr) {
    --prefix->attachments;
  }
  sequences_.erase(iterator);
  return Status::success();
}

Result<KvReservation>
PagedKvCache::reserve(SequenceId sequence_id,
                      std::uint32_t target_token_count) noexcept {
  const auto* sequence = find_sequence(sequence_id);
  if (sequence == nullptr ||
      find_sequence_reservation(sequence_id) != nullptr ||
      target_token_count < sequence->total_tokens ||
      next_reservation_id_ == 0) {
    return Result<KvReservation>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }

  const auto suffix_tokens = target_token_count - sequence->prefix_tokens;
  const auto required_pages = pages_for(suffix_tokens, tokens_per_page_);
  const auto existing_pages =
      static_cast<std::uint64_t>(sequence->suffix_pages.size());
  const auto new_page_count =
      required_pages > existing_pages ? required_pages - existing_pages : 0;

  try {
    std::vector<KvPageId> selected;
    selected.reserve(static_cast<std::size_t>(new_page_count));
    for (std::size_t index = 0;
         index < pages_.size() &&
         selected.size() < static_cast<std::size_t>(new_page_count);
         ++index) {
      if (pages_[index].state == PageState::free) {
        selected.push_back(static_cast<KvPageId>(index));
      }
    }
    if (selected.size() != static_cast<std::size_t>(new_page_count)) {
      return Result<KvReservation>::failure(
          Status::failure(ErrorCode::insufficient_memory));
    }

    const auto reservation_id = next_reservation_id_;
    KvReservation result{reservation_id, sequence_id, target_token_count,
                         selected};
    reservations_.push_back(PendingReservation{
        reservation_id,
        sequence_id,
        target_token_count,
        selected,
    });
    for (const auto page_id : selected) {
      pages_[page_id] = Page{PageState::reserved, reservation_id};
    }
    ++next_reservation_id_;
    return Result<KvReservation>::success(std::move(result));
  } catch (const std::bad_alloc&) {
    return Result<KvReservation>::failure(
        Status::failure(ErrorCode::allocation_failed));
  } catch (const std::length_error&) {
    return Result<KvReservation>::failure(
        Status::failure(ErrorCode::resource_limit));
  }
}

Status PagedKvCache::commit(ReservationId reservation_id) noexcept {
  auto* reservation = find_reservation(reservation_id);
  if (reservation == nullptr) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  auto* sequence = find_sequence(reservation->sequence_id);
  if (sequence == nullptr) {
    return Status::failure(ErrorCode::invalid_format);
  }
  for (const auto page_id : reservation->pages) {
    if (page_id >= pages_.size() ||
        pages_[page_id] !=
            Page{PageState::reserved, reservation_id}) {
      return Status::failure(ErrorCode::invalid_format);
    }
  }

  try {
    sequence->suffix_pages.reserve(sequence->suffix_pages.size() +
                                   reservation->pages.size());
  } catch (const std::bad_alloc&) {
    return Status::failure(ErrorCode::allocation_failed);
  } catch (const std::length_error&) {
    return Status::failure(ErrorCode::resource_limit);
  }

  sequence->suffix_pages.insert(sequence->suffix_pages.end(),
                                reservation->pages.begin(),
                                reservation->pages.end());
  sequence->total_tokens = reservation->target_token_count;
  for (const auto page_id : reservation->pages) {
    pages_[page_id] =
        Page{PageState::mutable_sequence, sequence->id};
  }
  const auto iterator = std::find_if(
      reservations_.begin(), reservations_.end(),
      [reservation_id](const PendingReservation& pending) {
        return pending.id == reservation_id;
      });
  reservations_.erase(iterator);
  return Status::success();
}

Status PagedKvCache::rollback(ReservationId reservation_id) noexcept {
  const auto iterator = std::find_if(
      reservations_.begin(), reservations_.end(),
      [reservation_id](const PendingReservation& reservation) {
        return reservation.id == reservation_id;
      });
  if (iterator == reservations_.end()) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  for (const auto page_id : iterator->pages) {
    if (page_id >= pages_.size() ||
        pages_[page_id] !=
            Page{PageState::reserved, reservation_id}) {
      return Status::failure(ErrorCode::invalid_format);
    }
  }
  for (const auto page_id : iterator->pages) {
    pages_[page_id] = Page{};
  }
  reservations_.erase(iterator);
  return Status::success();
}

Status PagedKvCache::publish_prefix(PrefixId prefix_id,
                                    SequenceId sequence_id) noexcept {
  auto* sequence = find_sequence(sequence_id);
  if (prefix_id == 0 || sequence == nullptr ||
      find_prefix(prefix_id) != nullptr || sequence->prefix_id != 0 ||
      sequence->total_tokens == 0 || sequence->suffix_pages.empty() ||
      find_sequence_reservation(sequence_id) != nullptr) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  for (const auto page_id : sequence->suffix_pages) {
    if (page_id >= pages_.size() ||
        pages_[page_id] !=
            Page{PageState::mutable_sequence, sequence_id}) {
      return Status::failure(ErrorCode::invalid_format);
    }
  }

  try {
    prefixes_.push_back(Prefix{
        prefix_id,
        sequence->total_tokens,
        1,
        sequence->suffix_pages,
    });
  } catch (const std::bad_alloc&) {
    return Status::failure(ErrorCode::allocation_failed);
  } catch (const std::length_error&) {
    return Status::failure(ErrorCode::resource_limit);
  }

  for (const auto page_id : sequence->suffix_pages) {
    pages_[page_id] = Page{PageState::immutable_prefix, prefix_id};
  }
  sequence->prefix_id = prefix_id;
  sequence->prefix_tokens = sequence->total_tokens;
  sequence->suffix_pages.clear();
  return Status::success();
}

Status PagedKvCache::destroy_prefix(PrefixId prefix_id) noexcept {
  const auto iterator =
      std::find_if(prefixes_.begin(), prefixes_.end(),
                   [prefix_id](const Prefix& prefix) {
                     return prefix.id == prefix_id;
                   });
  if (iterator == prefixes_.end() || iterator->attachments != 0) {
    return Status::failure(ErrorCode::invalid_argument);
  }
  for (const auto page_id : iterator->pages) {
    if (page_id >= pages_.size() ||
        pages_[page_id] !=
            Page{PageState::immutable_prefix, prefix_id}) {
      return Status::failure(ErrorCode::invalid_format);
    }
  }
  for (const auto page_id : iterator->pages) {
    pages_[page_id] = Page{};
  }
  prefixes_.erase(iterator);
  return Status::success();
}

Result<KvPageTable>
PagedKvCache::page_table(SequenceId sequence_id) const noexcept {
  const auto* sequence = find_sequence(sequence_id);
  if (sequence == nullptr) {
    return Result<KvPageTable>::failure(
        Status::failure(ErrorCode::invalid_argument));
  }
  const auto* prefix = sequence->prefix_id == 0
                           ? nullptr
                           : find_prefix(sequence->prefix_id);
  if (sequence->prefix_id != 0 && prefix == nullptr) {
    return Result<KvPageTable>::failure(
        Status::failure(ErrorCode::invalid_format));
  }

  try {
    KvPageTable result;
    result.sequence_id = sequence_id;
    result.prefix_id = sequence->prefix_id;
    result.prefix_tokens = sequence->prefix_tokens;
    result.total_tokens = sequence->total_tokens;
    result.shared_prefix_pages =
        prefix == nullptr
            ? 0U
            : static_cast<std::uint32_t>(prefix->pages.size());
    result.pages.reserve((prefix == nullptr ? 0U : prefix->pages.size()) +
                         sequence->suffix_pages.size());
    if (prefix != nullptr) {
      result.pages.insert(result.pages.end(), prefix->pages.begin(),
                          prefix->pages.end());
    }
    result.pages.insert(result.pages.end(),
                        sequence->suffix_pages.begin(),
                        sequence->suffix_pages.end());
    return Result<KvPageTable>::success(std::move(result));
  } catch (const std::bad_alloc&) {
    return Result<KvPageTable>::failure(
        Status::failure(ErrorCode::allocation_failed));
  } catch (const std::length_error&) {
    return Result<KvPageTable>::failure(
        Status::failure(ErrorCode::resource_limit));
  }
}

Result<KvCacheMetrics> PagedKvCache::metrics() const noexcept {
  KvCacheMetrics result;
  result.total_pages = pages_.size();
  result.sequence_count = sequences_.size();
  result.prefix_count = prefixes_.size();

  for (std::size_t page_id = 0; page_id < pages_.size(); ++page_id) {
    const auto& page = pages_[page_id];
    const auto typed_page_id = static_cast<KvPageId>(page_id);
    bool owner_is_valid = false;
    switch (page.state) {
    case PageState::free:
      ++result.free_pages;
      owner_is_valid = page.owner == 0;
      break;
    case PageState::reserved: {
      ++result.pending_pages;
      const auto* reservation = find_reservation(page.owner);
      owner_is_valid =
          reservation != nullptr &&
          std::find(reservation->pages.begin(), reservation->pages.end(),
                    typed_page_id) != reservation->pages.end();
      break;
    }
    case PageState::mutable_sequence: {
      ++result.mutable_pages;
      const auto* sequence = find_sequence(page.owner);
      owner_is_valid =
          sequence != nullptr &&
          std::find(sequence->suffix_pages.begin(),
                    sequence->suffix_pages.end(),
                    typed_page_id) != sequence->suffix_pages.end();
      break;
    }
    case PageState::immutable_prefix: {
      ++result.shared_prefix_pages;
      const auto* prefix = find_prefix(page.owner);
      owner_is_valid =
          prefix != nullptr &&
          std::find(prefix->pages.begin(), prefix->pages.end(),
                    typed_page_id) !=
              prefix->pages.end();
      break;
    }
    }
    if (!owner_is_valid) {
      return Result<KvCacheMetrics>::failure(
          Status::failure(ErrorCode::invalid_format));
    }
  }
  for (const auto& prefix : prefixes_) {
    if (!checked_add(result.prefix_attachments, prefix.attachments,
                     result.prefix_attachments)) {
      return Result<KvCacheMetrics>::failure(
          Status::failure(ErrorCode::arithmetic_overflow));
    }
  }
  for (const auto& sequence : sequences_) {
    if (!checked_add(result.logical_payload_tokens, sequence.total_tokens,
                     result.logical_payload_tokens)) {
      return Result<KvCacheMetrics>::failure(
          Status::failure(ErrorCode::arithmetic_overflow));
    }
    const auto* prefix =
        sequence.prefix_id == 0 ? nullptr : find_prefix(sequence.prefix_id);
    if (sequence.prefix_id != 0 && prefix == nullptr) {
      return Result<KvCacheMetrics>::failure(
          Status::failure(ErrorCode::invalid_format));
    }
    const auto logical_pages =
        sequence.suffix_pages.size() +
        (prefix == nullptr ? 0U : prefix->pages.size());
    std::uint64_t logical_capacity = 0;
    if (!checked_multiply(logical_pages, tokens_per_page_,
                          logical_capacity) ||
        !checked_add(result.logical_reserved_tokens, logical_capacity,
                     result.logical_reserved_tokens)) {
      return Result<KvCacheMetrics>::failure(
          Status::failure(ErrorCode::arithmetic_overflow));
    }
  }
  if (result.logical_payload_tokens > result.logical_reserved_tokens) {
    return Result<KvCacheMetrics>::failure(
        Status::failure(ErrorCode::invalid_format));
  }
  result.logical_fragmentation_tokens =
      result.logical_reserved_tokens - result.logical_payload_tokens;

  const auto committed_pages =
      result.mutable_pages + result.shared_prefix_pages;
  if (!checked_multiply(committed_pages, tokens_per_page_,
                        result.physical_committed_tokens)) {
    return Result<KvCacheMetrics>::failure(
        Status::failure(ErrorCode::arithmetic_overflow));
  }
  return Result<KvCacheMetrics>::success(result);
}

PagedKvCache::Sequence*
PagedKvCache::find_sequence(SequenceId id) noexcept {
  const auto iterator =
      std::find_if(sequences_.begin(), sequences_.end(),
                   [id](const Sequence& sequence) {
                     return sequence.id == id;
                   });
  return iterator == sequences_.end() ? nullptr : &*iterator;
}

const PagedKvCache::Sequence*
PagedKvCache::find_sequence(SequenceId id) const noexcept {
  const auto iterator =
      std::find_if(sequences_.begin(), sequences_.end(),
                   [id](const Sequence& sequence) {
                     return sequence.id == id;
                   });
  return iterator == sequences_.end() ? nullptr : &*iterator;
}

PagedKvCache::Prefix* PagedKvCache::find_prefix(PrefixId id) noexcept {
  const auto iterator =
      std::find_if(prefixes_.begin(), prefixes_.end(),
                   [id](const Prefix& prefix) {
                     return prefix.id == id;
                   });
  return iterator == prefixes_.end() ? nullptr : &*iterator;
}

const PagedKvCache::Prefix*
PagedKvCache::find_prefix(PrefixId id) const noexcept {
  const auto iterator =
      std::find_if(prefixes_.begin(), prefixes_.end(),
                   [id](const Prefix& prefix) {
                     return prefix.id == id;
                   });
  return iterator == prefixes_.end() ? nullptr : &*iterator;
}

PagedKvCache::PendingReservation*
PagedKvCache::find_reservation(ReservationId id) noexcept {
  const auto iterator =
      std::find_if(reservations_.begin(), reservations_.end(),
                   [id](const PendingReservation& reservation) {
                     return reservation.id == id;
                   });
  return iterator == reservations_.end() ? nullptr : &*iterator;
}

const PagedKvCache::PendingReservation*
PagedKvCache::find_reservation(ReservationId id) const noexcept {
  const auto iterator =
      std::find_if(reservations_.begin(), reservations_.end(),
                   [id](const PendingReservation& reservation) {
                     return reservation.id == id;
                   });
  return iterator == reservations_.end() ? nullptr : &*iterator;
}

const PagedKvCache::PendingReservation*
PagedKvCache::find_sequence_reservation(SequenceId id) const noexcept {
  const auto iterator =
      std::find_if(reservations_.begin(), reservations_.end(),
                   [id](const PendingReservation& reservation) {
                     return reservation.sequence_id == id;
                   });
  return iterator == reservations_.end() ? nullptr : &*iterator;
}

} // namespace marketforge::serving
