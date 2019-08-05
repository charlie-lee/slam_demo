# set target name
set(TARGET_NAME "system_vo")

# add executable
add_executable(${TARGET_NAME}
  ${TARGET_NAME}.cpp
  )

target_link_libraries(${TARGET_NAME}
  ${PROJECT_NAME} # SLAM demo lib
  ${OpenCV_LIBS} # OpenCV 4.0+
  )

target_compile_options(${TARGET_NAME} PRIVATE
  "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

set_target_properties(${TARGET_NAME}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_SOURCE_DIR}/examples/bin/${TARGET_NAME}"
  CXX_EXTENSIONS OFF
  )
