#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

#include "marketforge/workloads/radar_simulation.hpp"

namespace {

void usage(const char* executable) {
  std::cerr << "usage: " << executable << " [--output TRACE.json]\n";
}

} // namespace

int main(int argc, char** argv) {
  std::filesystem::path output_path;
  if (argc == 3 && std::string_view(argv[1]) == "--output") {
    output_path = argv[2];
  } else if (argc != 1) {
    usage(argv[0]);
    return 2;
  }

  auto result = marketforge::workloads::run_scenario();
  if (!result) {
    std::cerr << "scenario failed with error code "
              << static_cast<unsigned>(result.status().code) << '\n';
    return 1;
  }
  const auto& scenario = result.value();
  if (!output_path.empty()) {
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output ||
        !marketforge::workloads::write_trace_json(scenario, output)) {
      std::cerr << "could not write trace to " << output_path << '\n';
      return 1;
    }
  }

  const auto& metrics = scenario.metrics;
  std::cout << std::fixed << std::setprecision(3)
            << "{\"project\":\"market_sim\",\"result\":\"pass\""
            << ",\"decisions\":" << metrics.decisions
            << ",\"p50_latency_us\":" << metrics.p50_latency_microseconds
            << ",\"p95_latency_us\":" << metrics.p95_latency_microseconds
            << ",\"p99_latency_us\":" << metrics.p99_latency_microseconds
            << ",\"deadline_met_fraction\":" << metrics.deadline_met_fraction
            << ",\"grammar_valid_fraction\":" << metrics.grammar_valid_fraction
            << ",\"kv_page_reduction_fraction\":"
            << metrics.kv_page_reduction_fraction
            << ",\"faster_than_realtime_factor\":"
            << metrics.faster_than_realtime_factor << '}';
  if (!output_path.empty()) {
    std::cout << "\ntrace=" << output_path.string();
  }
  std::cout << '\n';
  return 0;
}
