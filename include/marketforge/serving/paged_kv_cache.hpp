#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "marketforge/core/result.hpp"
#include "marketforge/core/status.hpp"
#include "marketforge/serving/sequence_scheduler.hpp"

namespace marketforge::serving {

using KvPageId = std::uint32_t;
using PrefixId = std::uint64_t;
using ReservationId = std::uint64_t;

struct KvReservation {
  ReservationId id{0};
  SequenceId sequence_id{0};
  std::uint32_t target_token_count{0};
  std::vector<KvPageId> new_pages;
};

struct KvPageTable {
  SequenceId sequence_id{0};
  PrefixId prefix_id{0};
  std::uint32_t prefix_tokens{0};
  std::uint32_t total_tokens{0};
  std::uint32_t shared_prefix_pages{0};
  std::vector<KvPageId> pages;
};

struct KvCacheMetrics {
  std::uint64_t total_pages{0};
  std::uint64_t free_pages{0};
  std::uint64_t pending_pages{0};
  std::uint64_t mutable_pages{0};
  std::uint64_t shared_prefix_pages{0};
  std::uint64_t sequence_count{0};
  std::uint64_t prefix_count{0};
  std::uint64_t prefix_attachments{0};
  std::uint64_t logical_payload_tokens{0};
  std::uint64_t logical_reserved_tokens{0};
  std::uint64_t logical_fragmentation_tokens{0};
  std::uint64_t physical_committed_tokens{0};

  friend constexpr bool
  operator==(const KvCacheMetrics&, const KvCacheMetrics&) = default;
};

class PagedKvCache {
public:
  [[nodiscard]] static Result<PagedKvCache>
  create(std::uint32_t page_count,
         std::uint32_t tokens_per_page) noexcept;

  [[nodiscard]] Status create_sequence(SequenceId sequence_id,
                                       PrefixId prefix_id = 0) noexcept;
  [[nodiscard]] Status destroy_sequence(SequenceId sequence_id) noexcept;

  [[nodiscard]] Result<KvReservation>
  reserve(SequenceId sequence_id,
          std::uint32_t target_token_count) noexcept;
  [[nodiscard]] Status commit(ReservationId reservation_id) noexcept;
  [[nodiscard]] Status rollback(ReservationId reservation_id) noexcept;

  [[nodiscard]] Status publish_prefix(PrefixId prefix_id,
                                      SequenceId sequence_id) noexcept;
  [[nodiscard]] Status destroy_prefix(PrefixId prefix_id) noexcept;

  [[nodiscard]] Result<KvPageTable>
  page_table(SequenceId sequence_id) const noexcept;
  [[nodiscard]] Result<KvCacheMetrics> metrics() const noexcept;

  [[nodiscard]] std::uint32_t tokens_per_page() const noexcept {
    return tokens_per_page_;
  }

private:
  enum class PageState : std::uint8_t {
    free,
    reserved,
    mutable_sequence,
    immutable_prefix,
  };

  struct Page {
    PageState state{PageState::free};
    std::uint64_t owner{0};

    friend constexpr bool operator==(const Page&, const Page&) = default;
  };

  struct Sequence {
    SequenceId id{0};
    PrefixId prefix_id{0};
    std::uint32_t prefix_tokens{0};
    std::uint32_t total_tokens{0};
    std::vector<KvPageId> suffix_pages;
  };

  struct Prefix {
    PrefixId id{0};
    std::uint32_t token_count{0};
    std::uint64_t attachments{0};
    std::vector<KvPageId> pages;
  };

  struct PendingReservation {
    ReservationId id{0};
    SequenceId sequence_id{0};
    std::uint32_t target_token_count{0};
    std::vector<KvPageId> pages;
  };

  PagedKvCache(std::uint32_t tokens_per_page,
               std::vector<Page> pages) noexcept
      : pages_(std::move(pages)), tokens_per_page_(tokens_per_page) {}

  [[nodiscard]] Sequence* find_sequence(SequenceId id) noexcept;
  [[nodiscard]] const Sequence*
  find_sequence(SequenceId id) const noexcept;
  [[nodiscard]] Prefix* find_prefix(PrefixId id) noexcept;
  [[nodiscard]] const Prefix* find_prefix(PrefixId id) const noexcept;
  [[nodiscard]] PendingReservation*
  find_reservation(ReservationId id) noexcept;
  [[nodiscard]] const PendingReservation*
  find_reservation(ReservationId id) const noexcept;
  [[nodiscard]] const PendingReservation*
  find_sequence_reservation(SequenceId id) const noexcept;

  std::vector<Page> pages_;
  std::uint32_t tokens_per_page_{0};
  std::vector<Sequence> sequences_;
  std::vector<Prefix> prefixes_;
  std::vector<PendingReservation> reservations_;
  ReservationId next_reservation_id_{1};
};

} // namespace marketforge::serving
