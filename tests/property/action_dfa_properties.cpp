#include "test_support.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "marketforge/grammar/smollm2_market_actions.hpp"

namespace {

std::uint32_t ordinal(marketforge::Action action) {
  if (action.kind() == marketforge::ActionKind::hold) {
    return 0;
  }
  const auto within_side =
      static_cast<std::uint32_t>(action.quantity() - 1) * 99U +
      static_cast<std::uint32_t>(action.price_tick() - 1);
  return action.kind() == marketforge::ActionKind::buy_yes
             ? 1U + within_side
             : 1U + 8U * 99U + within_side;
}

MF_TEST(smollm2_catalog_exhaustively_decodes_every_canonical_action) {
  const auto catalog_result = marketforge::SmolLm2MarketActionCatalog::create();
  MF_CHECK(catalog_result);
  const auto& catalog = catalog_result.value();
  const auto& dfa = catalog.dfa();
  MF_CHECK_EQ(catalog.entries().size(), 1585);
  MF_CHECK_EQ(dfa.action_count(), 1585);
  MF_CHECK_EQ(dfa.state_count(), 3230);
  MF_CHECK_EQ(dfa.arc_count(), 3229);

  std::set<std::uint32_t> actions;
  for (const auto& entry : catalog.entries()) {
    auto state = dfa.root();
    MF_CHECK(!dfa.terminal(state));
    for (const auto token : entry.tokens) {
      const auto next = dfa.advance(state, token);
      MF_CHECK(next);
      state = next.value();
    }
    MF_CHECK(dfa.terminal(state));
    const auto decoded = dfa.decode_terminal(state);
    MF_CHECK(decoded);
    MF_CHECK_EQ(decoded.value(), entry.action);
    MF_CHECK(actions.insert(ordinal(decoded.value())).second);
    const auto canonical = catalog.tokens(entry.action);
    MF_CHECK(canonical);
    MF_CHECK(std::equal(canonical.value().begin(), canonical.value().end(),
                        entry.tokens.begin(), entry.tokens.end()));
  }
  MF_CHECK_EQ(actions.size(), 1585);
  MF_CHECK_EQ(*actions.begin(), 0);
  MF_CHECK_EQ(*actions.rbegin(), 1584);
}

MF_TEST(smollm2_dfa_graph_is_reachable_sorted_and_prefix_free) {
  const auto catalog_result = marketforge::SmolLm2MarketActionCatalog::create();
  MF_CHECK(catalog_result);
  const auto& dfa = catalog_result.value().dfa();
  std::vector<bool> reachable(dfa.state_count(), false);
  std::vector<marketforge::GrammarState> pending{dfa.root()};
  reachable[0] = true;
  std::size_t terminal_count = 0;
  std::size_t maximum_outgoing = 0;

  while (!pending.empty()) {
    const auto state = pending.back();
    pending.pop_back();
    const auto allowed = dfa.allowed(state);
    MF_CHECK(allowed);
    maximum_outgoing = std::max(maximum_outgoing, allowed.value().size());
    for (std::size_t index = 0; index < allowed.value().size(); ++index) {
      if (index != 0) {
        MF_CHECK(allowed.value()[index - 1].token <
                 allowed.value()[index].token);
      }
      const auto& arc = allowed.value()[index];
      MF_CHECK(arc.next.value < dfa.state_count());
      const auto transitioned = dfa.advance(state, arc.token);
      MF_CHECK(transitioned);
      MF_CHECK_EQ(transitioned.value(), arc.next);
      if (!reachable[arc.next.value]) {
        reachable[arc.next.value] = true;
        pending.push_back(arc.next);
      }
    }
    if (dfa.terminal(state)) {
      ++terminal_count;
      MF_CHECK(allowed.value().empty());
      MF_CHECK(dfa.decode_terminal(state));
    } else {
      MF_CHECK(!allowed.value().empty());
    }
  }

  MF_CHECK(std::all_of(reachable.begin(), reachable.end(),
                       [](bool value) { return value; }));
  MF_CHECK_EQ(terminal_count, 1585);
  MF_CHECK_EQ(maximum_outgoing, 11);
  MF_CHECK(!dfa.advance(
      dfa.root(), marketforge::SmolLm2MarketActionCatalog::vocabulary_size()));
}

} // namespace
