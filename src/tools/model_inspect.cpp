#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "marketforge/model/loaded_weights.hpp"
#include "marketforge/model/model_config.hpp"
#include "marketforge/model/model_spec.hpp"

int main(const int argument_count, const char* const* arguments) {
  if (argument_count != 3) {
    std::cerr << "usage: marketforge_model_inspect CONFIG WEIGHTS\n";
    return EXIT_FAILURE;
  }

  const auto config =
      marketforge::load_model_config(std::filesystem::path(arguments[1]));
  if (!config) {
    std::cerr << "config rejected; error code "
              << static_cast<unsigned>(config.status().code) << '\n';
    return EXIT_FAILURE;
  }
  if (config.value() != marketforge::smollm2_135m_profile().spec) {
    std::cerr << "config does not match the locked SmolLM2-135M contract\n";
    return EXIT_FAILURE;
  }

  const auto weights = marketforge::LoadedWeights::open_and_bind(
      std::filesystem::path(arguments[2]), config.value());
  if (!weights) {
    std::cerr << "weights rejected; error code "
              << static_cast<unsigned>(weights.status().code) << ", detail "
              << weights.status().detail << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "SmolLM2-135M metadata verified: "
            << weights.value().layers().size() << " layers, "
            << (weights.value().output_head_aliases_embedding()
                    ? "tied output embedding"
                    : "explicit output tensor")
            << '\n';
  return EXIT_SUCCESS;
}
