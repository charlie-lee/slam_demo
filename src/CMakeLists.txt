# add static shared lib
add_library(${PROJECT_NAME} SHARED
  System.cpp
  Config.cpp
  Tracker.cpp
  Frame.cpp
  )

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS} # OpenCV 4.0+
  )

target_compile_options(${PROJECT_NAME} PRIVATE
  "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_SOURCE_DIR}/lib"
  CXX_EXTENSIONS OFF
  )
