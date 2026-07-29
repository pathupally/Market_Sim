function(marketforge_enable_sanitizers target)
  if(NOT MARKETFORGE_ENABLE_SANITIZERS)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(
      ${target}
      PRIVATE
        -fno-omit-frame-pointer
        -fsanitize=address,undefined
    )
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  else()
    message(FATAL_ERROR "Sanitizers are not configured for ${CMAKE_CXX_COMPILER_ID}.")
  endif()
endfunction()
