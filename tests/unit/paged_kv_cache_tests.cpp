#include "test_support.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include "marketforge/core/status.hpp"
#include "marketforge/serving/paged_kv_cache.hpp"

namespace {

using marketforge::ErrorCode;
using marketforge::serving::KvCacheMetrics;
using marketforge::serving::KvPageTable;
using marketforge::serving::KvReservation;
using marketforge::serving::PagedKvCache;

PagedKvCache cache(std::uint32_t page_count = 8,
                   std::uint32_t tokens_per_page = 4) {
  auto result = PagedKvCache::create(page_count, tokens_per_page);
  MF_CHECK(result);
  return std::move(result).value();
}

KvReservation reserve(PagedKvCache& state, std::uint64_t sequence_id,
                      std::uint32_t target_tokens) {
  auto result = state.reserve(sequence_id, target_tokens);
  MF_CHECK(result);
  return std::move(result).value();
}

KvPageTable table(const PagedKvCache& state, std::uint64_t sequence_id) {
  auto result = state.page_table(sequence_id);
  MF_CHECK(result);
  return std::move(result).value();
}

KvCacheMetrics metrics(const PagedKvCache& state) {
  auto result = state.metrics();
  MF_CHECK(result);
  return result.value();
}

MF_TEST(paged_kv_cache_validates_creation_and_sequence_identity) {
  MF_CHECK_EQ(PagedKvCache::create(0, 4).status().code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(PagedKvCache::create(4, 0).status().code,
              ErrorCode::invalid_argument);

  auto state = cache();
  MF_CHECK_EQ(state.create_sequence(0).code, ErrorCode::invalid_argument);
  MF_CHECK_EQ(state.create_sequence(1, 99).code,
              ErrorCode::invalid_argument);
  MF_CHECK(state.create_sequence(1).ok());
  MF_CHECK_EQ(state.create_sequence(1).code, ErrorCode::invalid_argument);
  MF_CHECK_EQ(state.page_table(99).status().code,
              ErrorCode::invalid_argument);
}

MF_TEST(paged_kv_reservations_commit_or_rollback_atomically) {
  auto state = cache(5, 4);
  MF_CHECK(state.create_sequence(1).ok());

  const auto first = reserve(state, 1, 5);
  MF_CHECK_EQ(first.new_pages, (std::vector<std::uint32_t>{0, 1}));
  MF_CHECK_EQ(metrics(state).pending_pages, 2U);
  MF_CHECK(table(state, 1).pages.empty());
  MF_CHECK(state.commit(first.id).ok());
  MF_CHECK_EQ(table(state, 1).pages,
              (std::vector<std::uint32_t>{0, 1}));
  MF_CHECK_EQ(table(state, 1).total_tokens, 5U);
  MF_CHECK_EQ(metrics(state).logical_fragmentation_tokens, 3U);

  const auto within_capacity = reserve(state, 1, 8);
  MF_CHECK(within_capacity.new_pages.empty());
  MF_CHECK(state.commit(within_capacity.id).ok());
  MF_CHECK_EQ(metrics(state).logical_fragmentation_tokens, 0U);

  const auto rolled_back = reserve(state, 1, 9);
  MF_CHECK_EQ(rolled_back.new_pages,
              (std::vector<std::uint32_t>{2}));
  MF_CHECK(state.rollback(rolled_back.id).ok());
  MF_CHECK_EQ(table(state, 1).total_tokens, 8U);
  MF_CHECK_EQ(metrics(state).free_pages, 3U);
  MF_CHECK_EQ(state.commit(rolled_back.id).code,
              ErrorCode::invalid_argument);
}

MF_TEST(paged_kv_failed_reservation_preserves_all_state) {
  auto state = cache(2, 4);
  MF_CHECK(state.create_sequence(1).ok());
  const auto before = metrics(state);
  const auto failed = state.reserve(1, 9);
  MF_CHECK(!failed);
  MF_CHECK_EQ(failed.status().code, ErrorCode::insufficient_memory);
  MF_CHECK_EQ(metrics(state), before);
  MF_CHECK(table(state, 1).pages.empty());

  const auto pending = reserve(state, 1, 4);
  MF_CHECK_EQ(state.reserve(1, 4).status().code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(state.destroy_sequence(1).code,
              ErrorCode::invalid_argument);
  MF_CHECK(state.rollback(pending.id).ok());
  MF_CHECK(state.destroy_sequence(1).ok());
}

MF_TEST(paged_kv_shared_prefix_pages_are_immutable_and_reference_counted) {
  auto state = cache(8, 4);
  MF_CHECK(state.create_sequence(1).ok());
  const auto source = reserve(state, 1, 6);
  MF_CHECK(state.commit(source.id).ok());
  MF_CHECK(state.publish_prefix(100, 1).ok());

  auto source_table = table(state, 1);
  MF_CHECK_EQ(source_table.prefix_id, 100U);
  MF_CHECK_EQ(source_table.prefix_tokens, 6U);
  MF_CHECK_EQ(source_table.shared_prefix_pages, 2U);
  MF_CHECK_EQ(source_table.pages,
              (std::vector<std::uint32_t>{0, 1}));

  MF_CHECK(state.create_sequence(2, 100).ok());
  const auto branch = reserve(state, 2, 7);
  MF_CHECK_EQ(branch.new_pages, (std::vector<std::uint32_t>{2}));
  MF_CHECK(state.commit(branch.id).ok());
  const auto source_branch = reserve(state, 1, 7);
  MF_CHECK_EQ(source_branch.new_pages,
              (std::vector<std::uint32_t>{3}));
  MF_CHECK(state.commit(source_branch.id).ok());

  const auto branch_table = table(state, 2);
  MF_CHECK_EQ(branch_table.shared_prefix_pages, 2U);
  MF_CHECK_EQ(branch_table.pages,
              (std::vector<std::uint32_t>{0, 1, 2}));
  const auto accounting = metrics(state);
  MF_CHECK_EQ(accounting.prefix_attachments, 2U);
  MF_CHECK_EQ(accounting.logical_payload_tokens, 14U);
  MF_CHECK_EQ(accounting.logical_reserved_tokens, 24U);
  MF_CHECK_EQ(accounting.logical_fragmentation_tokens, 10U);
  MF_CHECK_EQ(accounting.shared_prefix_pages, 2U);
  MF_CHECK_EQ(accounting.mutable_pages, 2U);
  MF_CHECK_EQ(accounting.physical_committed_tokens, 16U);

  MF_CHECK_EQ(state.destroy_prefix(100).code,
              ErrorCode::invalid_argument);
  MF_CHECK(state.destroy_sequence(2).ok());
  MF_CHECK(state.destroy_sequence(1).ok());
  MF_CHECK(state.destroy_prefix(100).ok());
  MF_CHECK_EQ(metrics(state).free_pages, 8U);
}

MF_TEST(paged_kv_reuses_the_lowest_free_physical_pages) {
  auto state = cache(5, 2);
  MF_CHECK(state.create_sequence(1).ok());
  MF_CHECK(state.create_sequence(2).ok());
  auto first = reserve(state, 1, 4);
  auto second = reserve(state, 2, 4);
  MF_CHECK(state.commit(first.id).ok());
  MF_CHECK(state.commit(second.id).ok());
  MF_CHECK_EQ(table(state, 1).pages,
              (std::vector<std::uint32_t>{0, 1}));
  MF_CHECK_EQ(table(state, 2).pages,
              (std::vector<std::uint32_t>{2, 3}));

  MF_CHECK(state.destroy_sequence(1).ok());
  MF_CHECK(state.create_sequence(3).ok());
  const auto reused = reserve(state, 3, 3);
  MF_CHECK_EQ(reused.new_pages,
              (std::vector<std::uint32_t>{0, 1}));
}

MF_TEST(paged_kv_prefix_publication_rejects_aliases_and_pending_writes) {
  auto state = cache();
  MF_CHECK(state.create_sequence(1).ok());
  MF_CHECK_EQ(state.publish_prefix(1, 1).code,
              ErrorCode::invalid_argument);
  auto reservation = reserve(state, 1, 3);
  MF_CHECK_EQ(state.publish_prefix(1, 1).code,
              ErrorCode::invalid_argument);
  MF_CHECK(state.commit(reservation.id).ok());
  MF_CHECK(state.publish_prefix(1, 1).ok());
  MF_CHECK_EQ(state.publish_prefix(2, 1).code,
              ErrorCode::invalid_argument);
  MF_CHECK_EQ(state.create_sequence(2, 2).code,
              ErrorCode::invalid_argument);
}

MF_TEST(paged_kv_page_conservation_holds_under_repeated_churn) {
  auto state = cache(32, 4);
  std::vector<std::uint64_t> retained;
  for (std::uint64_t id = 1; id <= 100; ++id) {
    MF_CHECK(state.create_sequence(id).ok());
    const auto target = static_cast<std::uint32_t>((id * 7U) % 17U + 1U);
    const auto reservation = reserve(state, id, target);
    MF_CHECK(state.commit(reservation.id).ok());
    const auto accounting = metrics(state);
    MF_CHECK_EQ(accounting.free_pages + accounting.pending_pages +
                    accounting.mutable_pages +
                    accounting.shared_prefix_pages,
                accounting.total_pages);

    if (id % 3U == 0U) {
      retained.push_back(id);
      if (retained.size() > 4U) {
        MF_CHECK(state.destroy_sequence(retained.front()).ok());
        retained.erase(retained.begin());
      }
    } else {
      MF_CHECK(state.destroy_sequence(id).ok());
    }
  }
  for (const auto id : retained) {
    MF_CHECK(state.destroy_sequence(id).ok());
  }
  MF_CHECK_EQ(metrics(state).free_pages, 32U);
}

} // namespace
