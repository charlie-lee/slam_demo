# set minimum CMake version
cmake_minimum_required(VERSION 3.8...3.15)

# set cmake policy
if(${CMAKE_VERSION} VERSION_LESS 3.15)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()


# set up top-level project
project(slam_demo
  VERSION 0.5.0
  DESCRIPTION "SLAM demo"
  LANGUAGES CXX)


# set some variables
# compile options
# -Wshadow: warn the user if a variable declaration shadows another
#   with the same name in the same scope
# -Wnon-virtual-dtor: warn the user if a class with virtual functions has
#   a non-virtual destructor. This can lead to hard to track down memory errors.
set(DEBUG_OPTIONS "-Wall" "-Wextra" "-Wshadow" "-pedantic" "-Wnon-virtual-dtor")
set(RELEASE_OPTIONS "-Wall" "-Wextra" "-Wshadow" "-pedantic" "-Wnon-virtual-dtor"
  "-O3")
# custom OpenCV version
#set(OpenCV_VER 3.4.5)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# declare dependencies (must be called after project() is called!!!)
# OpenCV for vision algorithms
find_package(OpenCV 4.0 REQUIRED)
# custom OpenCV build: path should contain OpenCVConfig.cmake file!
#find_package(OpenCV ${OpenCV_VER} REQUIRED
#  PATHS "~/Workspace/c++/opencv/opencv-${OpenCV_VER}/build")
# Eigen3 for pose representations
find_package(Eigen3 3.3 REQUIRED)
# g2o for bundle adjustment
find_package(G2O REQUIRED)
# doxygen for documentation
find_package(Doxygen QUIET
  REQUIRED COMPONENTS dot)


# add sub directories
add_subdirectory(src)
add_subdirectory(examples)


# add document (basic version)
if(${DOXYGEN_FOUND})
  # 1) set doxygen-related variables
  #    - general options
  set(DOXYGEN_FILE_PATTERNS "*.c" "*.cpp" "*.h" "*.hpp" "*.md")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_EXTRACT_STATIC YES) # documenting static funcs
  set(DOXYGEN_EXTRACT_PRIVATE YES) # extract private members
  set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/doc)
  #    - math-related options
  set(DOXYGEN_USE_MATHJAX YES)
  set(DOXYGEN_MATHJAX_EXTENSIONS "TeX/AMSmath" "TeX/AMSsymbols")
  #set(DOXYGEN_EXTRA_PACKAGES amsmath)
  #    - README integration
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${PROJECT_SOURCE_DIR}/README.md")
  # 2) call cmake function to add doc
  doxygen_add_docs(doc
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/examples
    )
endif()
