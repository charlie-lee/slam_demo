# add static shared lib
add_library(${PROJECT_NAME} SHARED
  System.cpp
  Config.cpp
  Tracker.cpp
  Frame.cpp
  CamPose.cpp
  MapPoint.cpp
  Map.cpp
  Optimizer.cpp
  KeyFrame.cpp
  )

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS} # OpenCV 4.0+
  Eigen3::Eigen # Eigen3 3.3+
  # g2o-related libs
  g2o_core
  g2o_solver_cholmod
  g2o_solver_csparse
  g2o_types_sba
  )

target_compile_options(${PROJECT_NAME} PRIVATE
  "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>" # compile options in debug mode
  "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>" # release mode
  "-march=native" # essential for running g2o (on virtual machine?)
  ) 

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_SOURCE_DIR}/lib"
  CXX_EXTENSIONS OFF
  )
