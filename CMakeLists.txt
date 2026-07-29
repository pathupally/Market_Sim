cmake_minimum_required(VERSION 3.25)

project(
  MarketForge
  VERSION 0.1.0
  DESCRIPTION "Stateful-agent C++/CUDA inference runtime"
  LANGUAGES CXX
)

option(MARKETFORGE_ENABLE_CUDA "Build the CUDA backend" OFF)
option(MARKETFORGE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MARKETFORGE_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)

if(MARKETFORGE_ENABLE_CUDA)
  message(FATAL_ERROR "The CUDA backend begins in PR 6; PRs 1-5 remain CPU-only.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/MarketForgeWarnings.cmake)
include(cmake/MarketForgeSanitizers.cmake)

add_library(marketforge_core
  src/cpu/operators.cpp
  src/cpu/smollm2.cpp
  src/core/host_buffer.cpp
  src/core/mapped_file.cpp
  src/core/shape.cpp
  src/model/json_parser.cpp
  src/model/loaded_weights.cpp
  src/model/model_config.cpp
  src/model/model_spec.cpp
  src/model/safetensors.cpp
)
add_library(MarketForge::core ALIAS marketforge_core)

target_compile_features(marketforge_core PUBLIC cxx_std_20)
target_include_directories(
  marketforge_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
marketforge_set_warnings(marketforge_core)
marketforge_enable_sanitizers(marketforge_core)

add_library(marketforge_grammar
  src/grammar/action.cpp
  src/grammar/action_dfa.cpp
  src/grammar/smollm2_market_actions.cpp
)
add_library(MarketForge::grammar ALIAS marketforge_grammar)
target_compile_features(marketforge_grammar PUBLIC cxx_std_20)
target_include_directories(
  marketforge_grammar
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(marketforge_grammar PUBLIC MarketForge::core)
marketforge_set_warnings(marketforge_grammar)
marketforge_enable_sanitizers(marketforge_grammar)

add_library(marketforge_market
  src/market/batch_market.cpp
)
add_library(MarketForge::market ALIAS marketforge_market)
target_compile_features(marketforge_market PUBLIC cxx_std_20)
target_include_directories(
  marketforge_market
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(
  marketforge_market
  PUBLIC MarketForge::grammar
)
marketforge_set_warnings(marketforge_market)
marketforge_enable_sanitizers(marketforge_market)

if(APPLE)
  find_library(MARKETFORGE_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_compile_definitions(
    marketforge_core
    PRIVATE
      ACCELERATE_NEW_LAPACK
      MARKETFORGE_USE_ACCELERATE=1
  )
  target_link_libraries(
    marketforge_core
    PRIVATE "${MARKETFORGE_ACCELERATE_FRAMEWORK}"
  )
endif()

add_executable(marketforge_model_inspect src/tools/model_inspect.cpp)
target_link_libraries(marketforge_model_inspect PRIVATE MarketForge::core)
marketforge_set_warnings(marketforge_model_inspect)
marketforge_enable_sanitizers(marketforge_model_inspect)

add_executable(
  marketforge_smollm2_conformance
  src/tools/smollm2_conformance.cpp
)
target_link_libraries(
  marketforge_smollm2_conformance
  PRIVATE MarketForge::core
)
marketforge_set_warnings(marketforge_smollm2_conformance)
marketforge_enable_sanitizers(marketforge_smollm2_conformance)

add_executable(marketforge_market_trace src/tools/market_trace.cpp)
target_link_libraries(
  marketforge_market_trace
  PRIVATE MarketForge::market
)
marketforge_set_warnings(marketforge_market_trace)
marketforge_enable_sanitizers(marketforge_market_trace)

include(CTest)
if(BUILD_TESTING)
  add_executable(marketforge_unit_tests
    tests/unit/cpu_golden_tests.cpp
    tests/unit/core_tests.cpp
    tests/unit/cpu_operator_tests.cpp
    tests/unit/cpu_smollm2_tests.cpp
    tests/unit/action_dfa_tests.cpp
    tests/unit/loader_tests.cpp
    tests/unit/market_tests.cpp
    tests/unit/model_spec_tests.cpp
    tests/property/action_dfa_properties.cpp
    tests/property/market_properties.cpp
    tests/unit/test_main.cpp
  )
  target_link_libraries(
    marketforge_unit_tests
    PRIVATE MarketForge::market
  )
  target_include_directories(
    marketforge_unit_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/support"
  )
  target_compile_definitions(
    marketforge_unit_tests
    PRIVATE "MARKETFORGE_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\""
  )
  marketforge_set_warnings(marketforge_unit_tests)
  marketforge_enable_sanitizers(marketforge_unit_tests)

  add_test(NAME marketforge.unit COMMAND marketforge_unit_tests)
  add_test(NAME marketforge.market_trace COMMAND marketforge_market_trace)
endif()
