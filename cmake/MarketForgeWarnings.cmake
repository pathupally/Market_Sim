function(marketforge_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(MARKETFORGE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;-Wshadow>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra,-Wpedantic,-Wconversion,-Wsign-conversion,-Wshadow>"
    )
    if(MARKETFORGE_WARNINGS_AS_ERRORS)
      target_compile_options(
        ${target}
        PRIVATE
          "$<$<COMPILE_LANGUAGE:CXX>:-Werror>"
          "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Werror>"
      )
    endif()
  endif()
endfunction()
