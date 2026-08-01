#pragma once

#include <cstdint>
#include <iosfwd>
#include <string_view>
#include <vector>

#include "marketforge/core/result.hpp"

namespace velorum::autonomy {

struct Vec2 {
  double x{0.0};
  double y{0.0};

  friend constexpr bool operator==(const Vec2&, const Vec2&) = default;
};

enum class MissionAction : std::uint8_t {
  hold,
  investigate,
  intercept,
  evade,
  return_to_base,
};

[[nodiscard]] std::string_view action_name(MissionAction action) noexcept;

struct VehicleState {
  std::uint32_t id{0};
  Vec2 position;
  Vec2 velocity;
  double fuel_fraction{0.0};
  MissionAction action{MissionAction::hold};

  friend constexpr bool operator==(const VehicleState&,
                                   const VehicleState&) = default;
};

struct TargetState {
  std::uint32_t id{0};
  Vec2 position;
  Vec2 velocity;

  friend constexpr bool operator==(const TargetState&,
                                   const TargetState&) = default;
};

struct RadarReturn {
  std::uint32_t observer_id{0};
  std::uint32_t target_id{0};
  double true_range{0.0};
  double measured_range{0.0};
  double measured_bearing_radians{0.0};
  Vec2 estimated_position;

  friend constexpr bool operator==(const RadarReturn&,
                                   const RadarReturn&) = default;
};

struct DecisionTrace {
  std::uint64_t sequence_id{0};
  std::uint32_t vehicle_id{0};
  MissionAction action{MissionAction::hold};
  std::uint64_t arrival_microseconds{0};
  std::uint64_t completion_microseconds{0};
  std::uint64_t deadline_microseconds{0};
  std::uint64_t latency_microseconds{0};
  std::uint64_t final_batch_generation{0};
  bool deadline_met{false};
  bool grammar_valid{false};

  friend constexpr bool operator==(const DecisionTrace&,
                                   const DecisionTrace&) = default;
};

struct TraceFrame {
  std::uint32_t step{0};
  double simulation_seconds{0.0};
  std::vector<VehicleState> vehicles;
  std::vector<TargetState> targets;
  std::vector<RadarReturn> radar_returns;
  std::vector<DecisionTrace> decisions;

  friend bool operator==(const TraceFrame&, const TraceFrame&) = default;
};

struct WorkloadMetrics {
  std::uint64_t decisions{0};
  std::uint64_t radar_returns{0};
  std::uint64_t scheduler_batches{0};
  std::uint64_t prefill_batches{0};
  std::uint64_t decode_batches{0};
  std::uint64_t deadline_misses{0};
  std::uint64_t grammar_valid_decisions{0};
  std::uint64_t shared_prefix_tokens_reused{0};
  std::uint64_t peak_physical_kv_pages{0};
  std::uint64_t peak_unshared_kv_pages{0};
  std::uint32_t maximum_batch_size{0};
  double p50_latency_microseconds{0.0};
  double p95_latency_microseconds{0.0};
  double p99_latency_microseconds{0.0};
  double deadline_met_fraction{0.0};
  double grammar_valid_fraction{0.0};
  double jain_completion_fairness{0.0};
  double kv_page_reduction_fraction{0.0};
  double decisions_per_service_second{0.0};
  double service_busy_seconds{0.0};
  double simulated_seconds{0.0};
  double faster_than_realtime_factor{0.0};

  friend constexpr bool operator==(const WorkloadMetrics&,
                                   const WorkloadMetrics&) = default;
};

struct ScenarioConfig {
  std::uint32_t steps{48};
  std::uint32_t vehicle_count{32};
  std::uint32_t target_count{6};
  std::uint32_t maximum_batch_size{8};
  std::uint32_t prompt_tokens{68};
  std::uint32_t output_tokens{2};
  std::uint32_t shared_prefix_tokens{64};
  std::uint32_t kv_tokens_per_page{16};
  std::uint32_t kv_page_count{256};
  std::uint64_t frame_period_microseconds{20'000};
  std::uint64_t decision_deadline_microseconds{8'000};
  std::uint64_t seed{0x56454c4f52554dULL};
  double simulation_step_seconds{0.25};
  double world_half_extent{220.0};
  double radar_range{180.0};
  double radar_range_noise{0.75};
  double radar_bearing_noise_radians{0.006};

  friend constexpr bool operator==(const ScenarioConfig&,
                                   const ScenarioConfig&) = default;
};

struct ScenarioResult {
  ScenarioConfig config;
  WorkloadMetrics metrics;
  std::vector<TraceFrame> frames;

  friend bool operator==(const ScenarioResult&,
                         const ScenarioResult&) = default;
};

[[nodiscard]] marketforge::Result<ScenarioResult>
run_scenario(ScenarioConfig config = {}) noexcept;

[[nodiscard]] bool write_trace_json(const ScenarioResult& result,
                                    std::ostream& output);

} // namespace velorum::autonomy
