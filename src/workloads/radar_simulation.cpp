#include "marketforge/workloads/radar_simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <new>
#include <numbers>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "marketforge/core/status.hpp"
#include "marketforge/grammar/token_dfa.hpp"
#include "marketforge/serving/paged_kv_cache.hpp"
#include "marketforge/serving/sequence_scheduler.hpp"

namespace marketforge::workloads {
namespace {

using marketforge::ChoiceId;
using marketforge::EncodedChoice;
using marketforge::ErrorCode;
using marketforge::GrammarState;
using marketforge::Result;
using marketforge::Status;
using marketforge::token_id_t;
using marketforge::TokenDfa;
using marketforge::TokenDfaConfig;
using marketforge::serving::PagedKvCache;
using marketforge::serving::SequenceRequest;
using marketforge::serving::SequenceScheduler;
using marketforge::serving::WorkKind;

constexpr std::uint64_t prefix_id = 1;
constexpr std::uint64_t prefix_sequence_id = 1;
constexpr std::uint64_t first_decision_sequence_id = 1'000;
constexpr std::uint32_t action_vocabulary_size = 256;
constexpr std::array<std::array<token_id_t, 2>, 5> action_tokens{{
    {80, 11},
    {80, 23},
    {80, 37},
    {80, 41},
    {80, 59},
}};

class DeterministicNoise {
public:
  explicit DeterministicNoise(std::uint64_t seed) noexcept
      : state_(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed) {}

  [[nodiscard]] double symmetric() noexcept {
    state_ ^= state_ >> 12U;
    state_ ^= state_ << 25U;
    state_ ^= state_ >> 27U;
    const auto bits = state_ * 0x2545f4914f6cdd1dULL;
    const auto unit =
        static_cast<double>(bits >> 11U) * (1.0 / 9007199254740992.0);
    return unit * 2.0 - 1.0;
  }

private:
  std::uint64_t state_;
};

[[nodiscard]] Result<ScenarioResult> failure(ErrorCode code) {
  return Result<ScenarioResult>::failure(Status::failure(code));
}

[[nodiscard]] double magnitude(Vec2 value) noexcept {
  return std::hypot(value.x, value.y);
}

[[nodiscard]] Vec2 subtract(Vec2 left, Vec2 right) noexcept {
  return Vec2{left.x - right.x, left.y - right.y};
}

[[nodiscard]] Vec2 direction(Vec2 value) noexcept {
  const auto length = magnitude(value);
  return length <= std::numeric_limits<double>::epsilon()
             ? Vec2{}
             : Vec2{value.x / length, value.y / length};
}

[[nodiscard]] Vec2 scaled(Vec2 value, double scale) noexcept {
  return Vec2{value.x * scale, value.y * scale};
}

[[nodiscard]] std::uint64_t pages_for(std::uint32_t tokens,
                                      std::uint32_t page_size) noexcept {
  return (static_cast<std::uint64_t>(tokens) + page_size - 1U) / page_size;
}

[[nodiscard]] bool valid_config(const ScenarioConfig& config) noexcept {
  if (config.steps == 0 || config.vehicle_count == 0 ||
      config.target_count == 0 || config.maximum_batch_size == 0 ||
      config.output_tokens != 2 || config.shared_prefix_tokens == 0 ||
      config.prompt_tokens < config.shared_prefix_tokens ||
      config.prompt_tokens >
          std::numeric_limits<std::uint32_t>::max() - config.output_tokens ||
      config.kv_tokens_per_page == 0 || config.kv_page_count == 0 ||
      config.frame_period_microseconds == 0 ||
      config.decision_deadline_microseconds == 0 ||
      config.simulation_step_seconds <= 0.0 ||
      config.world_half_extent <= 0.0 || config.radar_range <= 0.0 ||
      config.radar_range_noise < 0.0 ||
      config.radar_bearing_noise_radians < 0.0 ||
      !std::isfinite(config.simulation_step_seconds) ||
      !std::isfinite(config.world_half_extent) ||
      !std::isfinite(config.radar_range) ||
      !std::isfinite(config.radar_range_noise) ||
      !std::isfinite(config.radar_bearing_noise_radians)) {
    return false;
  }
  const auto per_sequence_pages = pages_for(
      config.prompt_tokens + config.output_tokens - config.shared_prefix_tokens,
      config.kv_tokens_per_page);
  const auto prefix_pages =
      pages_for(config.shared_prefix_tokens, config.kv_tokens_per_page);
  return prefix_pages + per_sequence_pages * config.vehicle_count <=
         config.kv_page_count;
}

[[nodiscard]] Result<TokenDfa> build_action_dfa() noexcept {
  std::array<EncodedChoice, action_tokens.size()> language;
  for (std::size_t index = 0; index < language.size(); ++index) {
    language[index] = EncodedChoice{
        static_cast<ChoiceId>(index),
        {action_tokens[index][0], action_tokens[index][1]},
    };
  }
  return TokenDfa::build(language,
                         TokenDfaConfig{action_vocabulary_size, 2, 5, 8, 8});
}

[[nodiscard]] bool grammar_round_trip(const TokenDfa& dfa,
                                      MissionAction action) noexcept {
  const auto index = static_cast<std::size_t>(action);
  if (index >= action_tokens.size()) {
    return false;
  }
  GrammarState state = dfa.root();
  for (const auto token : action_tokens[index]) {
    const auto next = dfa.advance(state, token);
    if (!next) {
      return false;
    }
    state = next.value();
  }
  const auto decoded = dfa.decode_terminal(state);
  return decoded && decoded.value() == static_cast<ChoiceId>(index);
}

[[nodiscard]] std::vector<VehicleState>
initial_vehicles(const ScenarioConfig& config) {
  std::vector<VehicleState> result;
  result.reserve(config.vehicle_count);
  for (std::uint32_t index = 0; index < config.vehicle_count; ++index) {
    const auto angle = 2.0 * std::numbers::pi_v<double> *
                       static_cast<double>(index) /
                       static_cast<double>(config.vehicle_count);
    const auto ring = 34.0 + static_cast<double>(index % 4U) * 5.0;
    result.push_back(VehicleState{
        index + 1U,
        Vec2{ring * std::cos(angle), ring * std::sin(angle)},
        Vec2{-2.0 * std::sin(angle), 2.0 * std::cos(angle)},
        0.20 + 0.78 * static_cast<double>((index * 17U) % 31U) / 30.0,
        MissionAction::hold,
    });
  }
  return result;
}

[[nodiscard]] std::vector<TargetState>
initial_targets(const ScenarioConfig& config) {
  std::vector<TargetState> result;
  result.reserve(config.target_count);
  for (std::uint32_t index = 0; index < config.target_count; ++index) {
    const auto angle = 2.0 * std::numbers::pi_v<double> *
                       (static_cast<double>(index) + 0.35) /
                       static_cast<double>(config.target_count);
    const auto radius = 92.0 + static_cast<double>(index % 3U) * 31.0;
    result.push_back(TargetState{
        index + 1U,
        Vec2{radius * std::cos(angle), radius * std::sin(angle)},
        Vec2{5.0 * std::cos(angle + 1.1), 5.0 * std::sin(angle + 1.1)},
    });
  }
  return result;
}

[[nodiscard]] RadarReturn observe(const VehicleState& vehicle,
                                  const TargetState& target,
                                  const ScenarioConfig& config,
                                  DeterministicNoise& noise) noexcept {
  const auto relative = subtract(target.position, vehicle.position);
  const auto true_range = magnitude(relative);
  const auto measured_range =
      std::max(0.0, true_range + noise.symmetric() * config.radar_range_noise);
  const auto bearing = std::atan2(relative.y, relative.x) +
                       noise.symmetric() * config.radar_bearing_noise_radians;
  return RadarReturn{
      vehicle.id,
      target.id,
      true_range,
      measured_range,
      bearing,
      Vec2{vehicle.position.x + measured_range * std::cos(bearing),
           vehicle.position.y + measured_range * std::sin(bearing)},
  };
}

[[nodiscard]] MissionAction
choose_action(const VehicleState& vehicle,
              const RadarReturn* radar_return) noexcept {
  if (vehicle.fuel_fraction < 0.26) {
    return MissionAction::return_to_base;
  }
  if (radar_return == nullptr) {
    return MissionAction::hold;
  }
  if (radar_return->measured_range < 24.0) {
    return MissionAction::evade;
  }
  if (radar_return->measured_range < 92.0) {
    return MissionAction::intercept;
  }
  return MissionAction::investigate;
}

[[nodiscard]] double quantile(std::vector<std::uint64_t> values,
                              double probability) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(
      std::ceil(probability * static_cast<double>(values.size())));
  const auto index = std::min(values.size() - 1U, rank == 0 ? 0U : rank - 1U);
  return static_cast<double>(values[index]);
}

void reflect_axis(double extent, double& position, double& velocity) noexcept {
  if (position > extent) {
    position = extent - (position - extent);
    velocity = -std::abs(velocity);
  } else if (position < -extent) {
    position = -extent + (-extent - position);
    velocity = std::abs(velocity);
  }
}

void update_targets(std::vector<TargetState>& targets,
                    const ScenarioConfig& config) noexcept {
  for (auto& target : targets) {
    target.position.x += target.velocity.x * config.simulation_step_seconds;
    target.position.y += target.velocity.y * config.simulation_step_seconds;
    reflect_axis(config.world_half_extent, target.position.x,
                 target.velocity.x);
    reflect_axis(config.world_half_extent, target.position.y,
                 target.velocity.y);
  }
}

void update_vehicle(VehicleState& vehicle, const RadarReturn* radar_return,
                    const ScenarioConfig& config) noexcept {
  Vec2 desired{};
  double speed = 0.0;
  switch (vehicle.action) {
  case MissionAction::hold:
    desired = direction(vehicle.velocity);
    speed = 1.0;
    break;
  case MissionAction::investigate:
    desired = radar_return == nullptr
                  ? Vec2{}
                  : direction(subtract(radar_return->estimated_position,
                                       vehicle.position));
    speed = 7.0;
    break;
  case MissionAction::intercept:
    desired = radar_return == nullptr
                  ? Vec2{}
                  : direction(subtract(radar_return->estimated_position,
                                       vehicle.position));
    speed = 15.0;
    break;
  case MissionAction::evade:
    desired = radar_return == nullptr
                  ? Vec2{}
                  : direction(subtract(vehicle.position,
                                       radar_return->estimated_position));
    speed = 13.0;
    break;
  case MissionAction::return_to_base:
    desired = direction(scaled(vehicle.position, -1.0));
    speed = 10.0;
    break;
  }
  vehicle.velocity = scaled(desired, speed);
  vehicle.position.x += vehicle.velocity.x * config.simulation_step_seconds;
  vehicle.position.y += vehicle.velocity.y * config.simulation_step_seconds;
  reflect_axis(config.world_half_extent, vehicle.position.x,
               vehicle.velocity.x);
  reflect_axis(config.world_half_extent, vehicle.position.y,
               vehicle.velocity.y);
  const auto burn = 0.0015 + speed * 0.00008;
  vehicle.fuel_fraction = std::max(0.0, vehicle.fuel_fraction - burn);
}

[[nodiscard]] std::uint64_t
batch_duration_microseconds(WorkKind kind, std::uint32_t maximum_context_tokens,
                            std::size_t batch_size,
                            std::uint64_t generation) noexcept {
  const auto deterministic_jitter = (generation * 37U) % 101U;
  if (kind == WorkKind::prefill) {
    return 800U + static_cast<std::uint64_t>(maximum_context_tokens) * 7U +
           static_cast<std::uint64_t>(batch_size) * 35U + deterministic_jitter;
  }
  return 360U + static_cast<std::uint64_t>(batch_size) * 28U +
         deterministic_jitter;
}

void write_number(std::ostream& output, double value) {
  output << std::setprecision(10) << value;
}

void write_vec2(std::ostream& output, Vec2 value) {
  output << "{\"x\":";
  write_number(output, value.x);
  output << ",\"y\":";
  write_number(output, value.y);
  output << '}';
}

} // namespace

std::string_view action_name(MissionAction action) noexcept {
  switch (action) {
  case MissionAction::hold:
    return "HOLD";
  case MissionAction::investigate:
    return "INVESTIGATE";
  case MissionAction::intercept:
    return "INTERCEPT";
  case MissionAction::evade:
    return "EVADE";
  case MissionAction::return_to_base:
    return "RETURN";
  }
  return "UNKNOWN";
}

Result<ScenarioResult> run_scenario(ScenarioConfig config) noexcept {
  if (!valid_config(config)) {
    return failure(ErrorCode::invalid_argument);
  }
  try {
    auto dfa_result = build_action_dfa();
    auto cache_result =
        PagedKvCache::create(config.kv_page_count, config.kv_tokens_per_page);
    if (!dfa_result) {
      return failure(dfa_result.status().code);
    }
    if (!cache_result) {
      return failure(cache_result.status().code);
    }
    auto dfa = std::move(dfa_result).value();
    auto cache = std::move(cache_result).value();
    SequenceScheduler scheduler;
    if (!cache.create_sequence(prefix_sequence_id).ok()) {
      return failure(ErrorCode::invalid_format);
    }
    auto prefix_reservation =
        cache.reserve(prefix_sequence_id, config.shared_prefix_tokens);
    if (!prefix_reservation ||
        !cache.commit(prefix_reservation.value().id).ok() ||
        !cache.publish_prefix(prefix_id, prefix_sequence_id).ok()) {
      return failure(ErrorCode::invalid_format);
    }

    auto vehicles = initial_vehicles(config);
    auto targets = initial_targets(config);
    DeterministicNoise noise(config.seed);
    ScenarioResult result;
    result.config = config;
    result.frames.reserve(config.steps);
    std::vector<std::uint64_t> latencies;
    latencies.reserve(static_cast<std::size_t>(config.steps) *
                      config.vehicle_count);
    std::vector<std::uint64_t> completions(config.vehicle_count, 0);
    std::uint64_t service_cursor = 0;
    std::uint64_t busy_microseconds = 0;
    std::uint64_t next_sequence = first_decision_sequence_id;

    for (std::uint32_t step = 0; step < config.steps; ++step) {
      TraceFrame frame;
      frame.step = step;
      frame.simulation_seconds =
          static_cast<double>(step) * config.simulation_step_seconds;
      frame.vehicles = vehicles;
      frame.targets = targets;
      frame.radar_returns.reserve(config.vehicle_count);
      frame.decisions.reserve(config.vehicle_count);
      const auto arrival =
          static_cast<std::uint64_t>(step) * config.frame_period_microseconds;
      service_cursor = std::max(service_cursor, arrival);

      for (std::uint32_t index = 0; index < config.vehicle_count; ++index) {
        const auto& vehicle = vehicles[index];
        const TargetState* closest = nullptr;
        double closest_range = config.radar_range;
        for (const auto& target : targets) {
          const auto range =
              magnitude(subtract(target.position, vehicle.position));
          if (range <= closest_range) {
            closest = &target;
            closest_range = range;
          }
        }
        const RadarReturn* observation = nullptr;
        if (closest != nullptr) {
          frame.radar_returns.push_back(
              observe(vehicle, *closest, config, noise));
          observation = &frame.radar_returns.back();
        }
        const auto action = choose_action(vehicle, observation);
        const auto valid = grammar_round_trip(dfa, action);
        const auto sequence_id = next_sequence++;
        frame.decisions.push_back(DecisionTrace{
            sequence_id,
            vehicle.id,
            action,
            arrival,
            0,
            arrival + config.decision_deadline_microseconds,
            0,
            0,
            false,
            valid,
        });
        if (!valid ||
            !scheduler
                 .enqueue(SequenceRequest{sequence_id, config.prompt_tokens,
                                          config.output_tokens})
                 .ok() ||
            !cache.create_sequence(sequence_id, prefix_id).ok()) {
          return failure(ErrorCode::invalid_format);
        }
        auto reservation = cache.reserve(sequence_id, config.prompt_tokens +
                                                          config.output_tokens);
        if (!reservation || !cache.commit(reservation.value().id).ok()) {
          return failure(ErrorCode::insufficient_memory);
        }
      }

      const auto cache_metrics = cache.metrics();
      if (!cache_metrics) {
        return failure(cache_metrics.status().code);
      }
      const auto physical_pages = cache_metrics.value().mutable_pages +
                                  cache_metrics.value().shared_prefix_pages;
      result.metrics.peak_physical_kv_pages =
          std::max(result.metrics.peak_physical_kv_pages, physical_pages);
      const auto unshared_pages =
          pages_for(config.prompt_tokens + config.output_tokens,
                    config.kv_tokens_per_page) *
          config.vehicle_count;
      result.metrics.peak_unshared_kv_pages =
          std::max(result.metrics.peak_unshared_kv_pages, unshared_pages);

      std::uint32_t completed = 0;
      while (completed < config.vehicle_count) {
        auto batch_result = scheduler.schedule(config.maximum_batch_size);
        if (!batch_result || batch_result.value().items.empty()) {
          return failure(ErrorCode::invalid_format);
        }
        const auto& batch = batch_result.value();
        ++result.metrics.scheduler_batches;
        result.metrics.maximum_batch_size =
            std::max(result.metrics.maximum_batch_size,
                     static_cast<std::uint32_t>(batch.items.size()));
        const auto prefill_count = static_cast<std::size_t>(std::count_if(
            batch.items.begin(), batch.items.end(),
            [](const auto& item) { return item.work == WorkKind::prefill; }));
        const auto decode_count = batch.items.size() - prefill_count;
        if (prefill_count != 0) {
          ++result.metrics.prefill_batches;
        }
        if (decode_count != 0) {
          ++result.metrics.decode_batches;
        }
        std::uint32_t maximum_context = 0;
        for (const auto& item : batch.items) {
          maximum_context = std::max(maximum_context, item.context_tokens);
        }
        std::uint64_t duration = 0;
        if (prefill_count != 0) {
          duration +=
              batch_duration_microseconds(WorkKind::prefill, maximum_context,
                                          prefill_count, batch.generation);
        }
        if (decode_count != 0) {
          duration +=
              batch_duration_microseconds(WorkKind::decode, maximum_context,
                                          decode_count, batch.generation);
        }
        service_cursor += duration;
        busy_microseconds += duration;
        for (const auto& item : batch.items) {
          const auto terminal = item.work == WorkKind::decode;
          if (!scheduler.complete(item.id, 1, terminal).ok()) {
            return failure(ErrorCode::invalid_format);
          }
          if (!terminal) {
            continue;
          }
          const auto first_sequence = frame.decisions.front().sequence_id;
          if (item.id < first_sequence ||
              item.id >= first_sequence + frame.decisions.size()) {
            return failure(ErrorCode::invalid_format);
          }
          auto& decision = frame.decisions[static_cast<std::size_t>(
              item.id - first_sequence)];
          decision.completion_microseconds = service_cursor;
          decision.latency_microseconds = service_cursor - arrival;
          decision.final_batch_generation = batch.generation;
          decision.deadline_met =
              service_cursor <= decision.deadline_microseconds;
          latencies.push_back(decision.latency_microseconds);
          ++completions[decision.vehicle_id - 1U];
          result.metrics.deadline_misses += decision.deadline_met ? 0U : 1U;
          result.metrics.grammar_valid_decisions +=
              decision.grammar_valid ? 1U : 0U;
          if (!cache.destroy_sequence(item.id).ok()) {
            return failure(ErrorCode::invalid_format);
          }
          ++completed;
        }
      }

      result.metrics.radar_returns += frame.radar_returns.size();
      result.metrics.decisions += frame.decisions.size();
      result.metrics.shared_prefix_tokens_reused +=
          static_cast<std::uint64_t>(config.shared_prefix_tokens) *
          frame.decisions.size();
      for (std::size_t index = 0; index < vehicles.size(); ++index) {
        vehicles[index].action = frame.decisions[index].action;
        const RadarReturn* observation = nullptr;
        const auto iterator =
            std::find_if(frame.radar_returns.begin(), frame.radar_returns.end(),
                         [id = vehicles[index].id](const RadarReturn& value) {
                           return value.observer_id == id;
                         });
        if (iterator != frame.radar_returns.end()) {
          observation = &*iterator;
        }
        frame.vehicles[index].action = vehicles[index].action;
        update_vehicle(vehicles[index], observation, config);
      }
      update_targets(targets, config);
      result.frames.push_back(std::move(frame));
    }

    if (!cache.destroy_sequence(prefix_sequence_id).ok() ||
        !cache.destroy_prefix(prefix_id).ok()) {
      return failure(ErrorCode::invalid_format);
    }
    const auto final_cache_metrics = cache.metrics();
    if (!final_cache_metrics ||
        final_cache_metrics.value().free_pages != config.kv_page_count) {
      return failure(ErrorCode::invalid_format);
    }

    result.metrics.p50_latency_microseconds = quantile(latencies, 0.50);
    result.metrics.p95_latency_microseconds = quantile(latencies, 0.95);
    result.metrics.p99_latency_microseconds = quantile(latencies, 0.99);
    const auto decisions = static_cast<double>(result.metrics.decisions);
    result.metrics.deadline_met_fraction =
        decisions == 0.0
            ? 0.0
            : 1.0 - static_cast<double>(result.metrics.deadline_misses) /
                        decisions;
    result.metrics.grammar_valid_fraction =
        decisions == 0.0
            ? 0.0
            : static_cast<double>(result.metrics.grammar_valid_decisions) /
                  decisions;
    double completion_sum = 0.0;
    double completion_square_sum = 0.0;
    for (const auto count : completions) {
      const auto value = static_cast<double>(count);
      completion_sum += value;
      completion_square_sum += value * value;
    }
    result.metrics.jain_completion_fairness =
        completion_square_sum == 0.0
            ? 0.0
            : completion_sum * completion_sum /
                  (static_cast<double>(completions.size()) *
                   completion_square_sum);
    result.metrics.kv_page_reduction_fraction =
        result.metrics.peak_unshared_kv_pages == 0
            ? 0.0
            : 1.0 - static_cast<double>(result.metrics.peak_physical_kv_pages) /
                        static_cast<double>(
                            result.metrics.peak_unshared_kv_pages);
    result.metrics.service_busy_seconds =
        static_cast<double>(busy_microseconds) / 1'000'000.0;
    result.metrics.simulated_seconds =
        static_cast<double>(config.steps) * config.simulation_step_seconds;
    result.metrics.decisions_per_service_second =
        result.metrics.service_busy_seconds == 0.0
            ? 0.0
            : decisions / result.metrics.service_busy_seconds;
    result.metrics.faster_than_realtime_factor =
        result.metrics.service_busy_seconds == 0.0
            ? 0.0
            : result.metrics.simulated_seconds /
                  result.metrics.service_busy_seconds;
    return Result<ScenarioResult>::success(std::move(result));
  } catch (const std::bad_alloc&) {
    return failure(ErrorCode::allocation_failed);
  } catch (const std::length_error&) {
    return failure(ErrorCode::resource_limit);
  }
}

bool write_trace_json(const ScenarioResult& result, std::ostream& output) {
  output << "{\"schema_version\":1,\"project\":\"market_sim\",\"scenario\":{";
  output << "\"steps\":" << result.config.steps
         << ",\"vehicle_count\":" << result.config.vehicle_count
         << ",\"target_count\":" << result.config.target_count << ",\"seed\":\""
         << result.config.seed << '"' << ",\"simulation_step_seconds\":";
  write_number(output, result.config.simulation_step_seconds);
  output << ",\"frame_period_microseconds\":"
         << result.config.frame_period_microseconds
         << ",\"decision_deadline_microseconds\":"
         << result.config.decision_deadline_microseconds
         << ",\"maximum_batch_size\":" << result.config.maximum_batch_size
         << ",\"world_half_extent\":";
  write_number(output, result.config.world_half_extent);
  output << ",\"radar_range\":";
  write_number(output, result.config.radar_range);
  output << "},\"metrics\":{";
  const auto& metrics = result.metrics;
  output << "\"decisions\":" << metrics.decisions
         << ",\"radar_returns\":" << metrics.radar_returns
         << ",\"scheduler_batches\":" << metrics.scheduler_batches
         << ",\"prefill_batches\":" << metrics.prefill_batches
         << ",\"decode_batches\":" << metrics.decode_batches
         << ",\"maximum_batch_size\":" << metrics.maximum_batch_size
         << ",\"deadline_misses\":" << metrics.deadline_misses
         << ",\"grammar_valid_decisions\":" << metrics.grammar_valid_decisions
         << ",\"shared_prefix_tokens_reused\":"
         << metrics.shared_prefix_tokens_reused
         << ",\"peak_physical_kv_pages\":" << metrics.peak_physical_kv_pages
         << ",\"peak_unshared_kv_pages\":" << metrics.peak_unshared_kv_pages
         << ",\"p50_latency_microseconds\":";
  write_number(output, metrics.p50_latency_microseconds);
  output << ",\"p95_latency_microseconds\":";
  write_number(output, metrics.p95_latency_microseconds);
  output << ",\"p99_latency_microseconds\":";
  write_number(output, metrics.p99_latency_microseconds);
  output << ",\"deadline_met_fraction\":";
  write_number(output, metrics.deadline_met_fraction);
  output << ",\"grammar_valid_fraction\":";
  write_number(output, metrics.grammar_valid_fraction);
  output << ",\"jain_completion_fairness\":";
  write_number(output, metrics.jain_completion_fairness);
  output << ",\"kv_page_reduction_fraction\":";
  write_number(output, metrics.kv_page_reduction_fraction);
  output << ",\"decisions_per_service_second\":";
  write_number(output, metrics.decisions_per_service_second);
  output << ",\"service_busy_seconds\":";
  write_number(output, metrics.service_busy_seconds);
  output << ",\"simulated_seconds\":";
  write_number(output, metrics.simulated_seconds);
  output << ",\"faster_than_realtime_factor\":";
  write_number(output, metrics.faster_than_realtime_factor);
  output << "},\"frames\":[";
  for (std::size_t frame_index = 0; frame_index < result.frames.size();
       ++frame_index) {
    if (frame_index != 0) {
      output << ',';
    }
    const auto& frame = result.frames[frame_index];
    output << "{\"step\":" << frame.step << ",\"simulation_seconds\":";
    write_number(output, frame.simulation_seconds);
    output << ",\"vehicles\":[";
    for (std::size_t index = 0; index < frame.vehicles.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      const auto& vehicle = frame.vehicles[index];
      output << "{\"id\":" << vehicle.id << ",\"position\":";
      write_vec2(output, vehicle.position);
      output << ",\"velocity\":";
      write_vec2(output, vehicle.velocity);
      output << ",\"fuel_fraction\":";
      write_number(output, vehicle.fuel_fraction);
      output << ",\"action\":\"" << action_name(vehicle.action) << "\"}";
    }
    output << "],\"targets\":[";
    for (std::size_t index = 0; index < frame.targets.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      const auto& target = frame.targets[index];
      output << "{\"id\":" << target.id << ",\"position\":";
      write_vec2(output, target.position);
      output << ",\"velocity\":";
      write_vec2(output, target.velocity);
      output << '}';
    }
    output << "],\"radar_returns\":[";
    for (std::size_t index = 0; index < frame.radar_returns.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      const auto& value = frame.radar_returns[index];
      output << "{\"observer_id\":" << value.observer_id
             << ",\"target_id\":" << value.target_id << ",\"true_range\":";
      write_number(output, value.true_range);
      output << ",\"measured_range\":";
      write_number(output, value.measured_range);
      output << ",\"measured_bearing_radians\":";
      write_number(output, value.measured_bearing_radians);
      output << ",\"estimated_position\":";
      write_vec2(output, value.estimated_position);
      output << '}';
    }
    output << "],\"decisions\":[";
    for (std::size_t index = 0; index < frame.decisions.size(); ++index) {
      if (index != 0) {
        output << ',';
      }
      const auto& decision = frame.decisions[index];
      output << "{\"sequence_id\":" << decision.sequence_id
             << ",\"vehicle_id\":" << decision.vehicle_id << ",\"action\":\""
             << action_name(decision.action)
             << "\",\"arrival_microseconds\":" << decision.arrival_microseconds
             << ",\"completion_microseconds\":"
             << decision.completion_microseconds
             << ",\"deadline_microseconds\":" << decision.deadline_microseconds
             << ",\"latency_microseconds\":" << decision.latency_microseconds
             << ",\"final_batch_generation\":"
             << decision.final_batch_generation << ",\"deadline_met\":"
             << (decision.deadline_met ? "true" : "false")
             << ",\"grammar_valid\":"
             << (decision.grammar_valid ? "true" : "false") << '}';
    }
    output << "]}";
  }
  output << "]}\n";
  return output.good();
}

} // namespace marketforge::workloads
