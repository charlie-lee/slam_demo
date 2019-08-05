# add executable
add_executable(${TARGET_MONO_VO}
  ${TARGET_MONO_VO}.cpp
  )

target_link_libraries(${TARGET_MONO_VO}
  ${PROJECT_NAME} # SLAM demo lib
  ${OpenCV_LIBS} # OpenCV 4.0+
  )

target_compile_options(${TARGET_MONO_VO} PRIVATE
  "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

set_target_properties(${TARGET_MONO_VO}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_SOURCE_DIR}/examples/bin/${TARGET_MONO_VO}"
  CXX_EXTENSIONS OFF
  )
