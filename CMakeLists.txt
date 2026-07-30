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

if(MARKETFORGE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(marketforge_cuda
    src/cuda/cublas_handle.cu
    src/cuda/cuda_stream.cu
    src/cuda/device_buffer.cu
    src/cuda/linear.cu
    src/cuda/lifecycle_probe.cu
    src/cuda/rms_norm.cu
  )
  add_library(MarketForge::cuda ALIAS marketforge_cuda)
  target_compile_features(marketforge_cuda PUBLIC cxx_std_20 cuda_std_20)
  set_target_properties(
    marketforge_cuda
    PROPERTIES
      CUDA_ARCHITECTURES 89
      CUDA_EXTENSIONS OFF
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
  )
  target_include_directories(
    marketforge_cuda
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"
  )
  target_link_libraries(
    marketforge_cuda
    PRIVATE
      CUDA::cublas
      CUDA::cudart
  )
  marketforge_set_warnings(marketforge_cuda)
endif()

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
  add_test(
    NAME marketforge.warning_policy
    COMMAND
      "${CMAKE_COMMAND}"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/warning_policy.cmake"
  )

  add_executable(
    marketforge_cuda_public_headers
    tests/compile/cuda_public_headers.cpp
  )
  target_include_directories(
    marketforge_cuda_public_headers
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_features(marketforge_cuda_public_headers PRIVATE cxx_std_20)
  marketforge_set_warnings(marketforge_cuda_public_headers)
  add_test(
    NAME marketforge.cuda_public_headers
    COMMAND marketforge_cuda_public_headers
  )

  if(MARKETFORGE_ENABLE_CUDA)
    add_executable(
      marketforge_cuda_probe
      src/tools/cuda_probe.cpp
    )
    target_link_libraries(
      marketforge_cuda_probe
      PRIVATE MarketForge::cuda
    )
    marketforge_set_warnings(marketforge_cuda_probe)

    add_executable(
      marketforge_cuda_unit_tests
      tests/cuda/cuda_tests.cu
      tests/unit/test_main.cpp
    )
    target_link_libraries(
      marketforge_cuda_unit_tests
      PRIVATE
        MarketForge::core
        MarketForge::cuda
    )
    target_include_directories(
      marketforge_cuda_unit_tests
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/support"
    )
    target_compile_features(
      marketforge_cuda_unit_tests
      PRIVATE cxx_std_20 cuda_std_20
    )
    set_target_properties(
      marketforge_cuda_unit_tests
      PROPERTIES
        CUDA_ARCHITECTURES 89
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    marketforge_set_warnings(marketforge_cuda_unit_tests)

    add_executable(
      marketforge_cuda_sanitizer_canary
      tests/cuda/sanitizer_canary.cu
    )
    target_link_libraries(
      marketforge_cuda_sanitizer_canary
      PRIVATE CUDA::cudart
    )
    target_compile_features(
      marketforge_cuda_sanitizer_canary
      PRIVATE cxx_std_20 cuda_std_20
    )
    set_target_properties(
      marketforge_cuda_sanitizer_canary
      PROPERTIES
        CUDA_ARCHITECTURES 89
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    marketforge_set_warnings(marketforge_cuda_sanitizer_canary)

    add_executable(
      marketforge_cuda_rmsnorm_bench
      src/tools/cuda_rmsnorm_bench.cu
    )
    target_link_libraries(
      marketforge_cuda_rmsnorm_bench
      PRIVATE
        MarketForge::cuda
        CUDA::cudart
    )
    target_compile_features(
      marketforge_cuda_rmsnorm_bench
      PRIVATE cxx_std_20 cuda_std_20
    )
    set_target_properties(
      marketforge_cuda_rmsnorm_bench
      PROPERTIES
        CUDA_ARCHITECTURES 89
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    marketforge_set_warnings(marketforge_cuda_rmsnorm_bench)

    add_executable(
      marketforge_cuda_linear_bench
      src/tools/cuda_linear_bench.cu
    )
    target_link_libraries(
      marketforge_cuda_linear_bench
      PRIVATE
        MarketForge::cuda
        CUDA::cudart
    )
    target_compile_features(
      marketforge_cuda_linear_bench
      PRIVATE cxx_std_20 cuda_std_20
    )
    set_target_properties(
      marketforge_cuda_linear_bench
      PROPERTIES
        CUDA_ARCHITECTURES 89
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    marketforge_set_warnings(marketforge_cuda_linear_bench)

    add_test(NAME marketforge.cuda.unit COMMAND marketforge_cuda_unit_tests)
    add_test(
      NAME marketforge.cuda.probe
      COMMAND marketforge_cuda_probe --repetitions 1
    )
  endif()
endif()
