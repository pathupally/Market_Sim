include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/MarketForgeWarnings.cmake")

foreach(
  expected
  IN ITEMS
    -Wall
    -Wextra
    -Wconversion
    -Wsign-conversion
    -Wshadow
)
  if(NOT expected IN_LIST _marketforge_cuda_host_warnings)
    message(FATAL_ERROR "CUDA host warning policy omits ${expected}")
  endif()
endforeach()

if(-Wpedantic IN_LIST _marketforge_cuda_host_warnings)
  message(
    FATAL_ERROR
    "CUDA host warning policy must not reject NVCC-generated GNU linemarkers"
  )
endif()

if(NOT -Wpedantic IN_LIST _marketforge_cxx_warnings)
  message(FATAL_ERROR "ordinary C++ compilation must retain -Wpedantic")
endif()

if(NOT _marketforge_cuda_frontend_werror STREQUAL "--Werror=all-warnings")
  message(FATAL_ERROR "NVCC frontend warnings must remain errors")
endif()

if(NOT _marketforge_cuda_host_werror STREQUAL "-Werror")
  message(FATAL_ERROR "CUDA host-compiler warnings must remain errors")
endif()
