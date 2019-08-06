# add static shared lib
add_library(${PROJECT_NAME} SHARED
  System.cpp
  )

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_SOURCE_DIR}/lib"
  CXX_EXTENSIONS OFF
  )
