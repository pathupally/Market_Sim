set(
  _marketforge_cxx_warnings
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wsign-conversion
  -Wshadow
)

# NVCC emits GNU linemarkers in its generated cudafe C++ source. GCC 13
# diagnoses those linemarkers under -Wpedantic before it reaches project
# source, and that diagnostic has no narrower warning switch. Keep the
# actionable host warnings strict here, and make NVCC's own diagnostics fatal
# separately with --Werror=all-warnings below.
set(
  _marketforge_cuda_host_warnings
  -Wall
  -Wextra
  -Wconversion
  -Wsign-conversion
  -Wshadow
)
set(_marketforge_cuda_frontend_werror --Werror=all-warnings)
set(_marketforge_cuda_host_werror -Werror)

function(marketforge_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(MARKETFORGE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    list(JOIN _marketforge_cuda_host_warnings "," cuda_host_warnings)
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${_marketforge_cxx_warnings}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${cuda_host_warnings}>"
    )
    if(MARKETFORGE_WARNINGS_AS_ERRORS)
      target_compile_options(
        ${target}
        PRIVATE
          "$<$<COMPILE_LANGUAGE:CXX>:-Werror>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${_marketforge_cuda_frontend_werror}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${_marketforge_cuda_host_werror}>"
      )
    endif()
  endif()
endfunction()
