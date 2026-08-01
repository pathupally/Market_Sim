#include "test_support.hpp"

#include <cmath>
#include <sstream>

#include "marketforge/core/status.hpp"
#include "marketforge/workloads/radar_simulation.hpp"

namespace {

using marketforge::ErrorCode;
using marketforge::workloads::action_name;
using marketforge::workloads::MissionAction;
using marketforge::workloads::run_scenario;
using marketforge::workloads::ScenarioConfig;
using marketforge::workloads::write_trace_json;

ScenarioConfig compact_config() {
  ScenarioConfig config;
  config.steps = 12;
  config.vehicle_count = 10;
  config.target_count = 3;
  config.maximum_batch_size = 4;
  config.kv_page_count = 64;
  return config;
}

MF_TEST(radar_workload_rejects_invalid_or_underprovisioned_scenarios) {
  auto config = compact_config();
  config.vehicle_count = 0;
  MF_CHECK_EQ(run_scenario(config).status().code, ErrorCode::invalid_argument);

  config = compact_config();
  config.output_tokens = 3;
  MF_CHECK_EQ(run_scenario(config).status().code, ErrorCode::invalid_argument);

  config = compact_config();
  config.kv_page_count = 4;
  MF_CHECK_EQ(run_scenario(config).status().code, ErrorCode::invalid_argument);
}

MF_TEST(radar_workload_is_deterministic_and_grammar_safe) {
  const auto config = compact_config();
  const auto first = run_scenario(config);
  const auto second = run_scenario(config);
  MF_CHECK(first);
  MF_CHECK(second);
  MF_CHECK_EQ(first.value(), second.value());
  MF_CHECK_EQ(first.value().frames.size(), config.steps);
  MF_CHECK_EQ(first.value().metrics.decisions,
              static_cast<std::uint64_t>(config.steps) * config.vehicle_count);
  MF_CHECK_EQ(first.value().metrics.grammar_valid_decisions,
              first.value().metrics.decisions);
  MF_CHECK_NEAR(first.value().metrics.grammar_valid_fraction, 1.0, 0.0);
  for (const auto& frame : first.value().frames) {
    MF_CHECK_EQ(frame.vehicles.size(), config.vehicle_count);
    MF_CHECK_EQ(frame.targets.size(), config.target_count);
    MF_CHECK_EQ(frame.decisions.size(), config.vehicle_count);
    for (const auto& decision : frame.decisions) {
      MF_CHECK(decision.grammar_valid);
      MF_CHECK(decision.completion_microseconds >=
               decision.arrival_microseconds);
      MF_CHECK_EQ(decision.latency_microseconds,
                  decision.completion_microseconds -
                      decision.arrival_microseconds);
    }
  }
}

MF_TEST(radar_workload_exercises_batching_deadlines_and_shared_prefixes) {
  const auto result = run_scenario(compact_config());
  MF_CHECK(result);
  const auto& metrics = result.value().metrics;
  MF_CHECK(metrics.scheduler_batches > 0U);
  MF_CHECK(metrics.prefill_batches > 0U);
  MF_CHECK(metrics.decode_batches > 0U);
  MF_CHECK_EQ(metrics.maximum_batch_size, 4U);
  MF_CHECK(metrics.p50_latency_microseconds > 0.0);
  MF_CHECK(metrics.p95_latency_microseconds >=
           metrics.p50_latency_microseconds);
  MF_CHECK(metrics.p99_latency_microseconds >=
           metrics.p95_latency_microseconds);
  MF_CHECK(metrics.deadline_met_fraction > 0.0);
  MF_CHECK(metrics.deadline_met_fraction <= 1.0);
  MF_CHECK_NEAR(metrics.jain_completion_fairness, 1.0, 1.0e-12);
  MF_CHECK(metrics.shared_prefix_tokens_reused > 0U);
  MF_CHECK(metrics.peak_physical_kv_pages < metrics.peak_unshared_kv_pages);
  MF_CHECK(metrics.kv_page_reduction_fraction > 0.0);
  MF_CHECK(metrics.decisions_per_service_second > 0.0);
  MF_CHECK(metrics.faster_than_realtime_factor > 1.0);
}

MF_TEST(radar_workload_trace_json_contains_replay_contract) {
  const auto result = run_scenario(compact_config());
  MF_CHECK(result);
  std::ostringstream output;
  MF_CHECK(write_trace_json(result.value(), output));
  const auto trace = output.str();
  MF_CHECK(
      trace.starts_with("{\"schema_version\":1,\"project\":\"market_sim\""));
  MF_CHECK(trace.find("\"radar_returns\":[") != std::string::npos);
  MF_CHECK(trace.find("\"grammar_valid\":true") != std::string::npos);
  MF_CHECK(trace.find("\"metrics\":{") != std::string::npos);
  MF_CHECK(trace.ends_with("}\n"));
}

MF_TEST(radar_workload_action_labels_are_stable) {
  MF_CHECK_EQ(action_name(MissionAction::hold), "HOLD");
  MF_CHECK_EQ(action_name(MissionAction::investigate), "INVESTIGATE");
  MF_CHECK_EQ(action_name(MissionAction::intercept), "INTERCEPT");
  MF_CHECK_EQ(action_name(MissionAction::evade), "EVADE");
  MF_CHECK_EQ(action_name(MissionAction::return_to_base), "RETURN");
}

} // namespace
