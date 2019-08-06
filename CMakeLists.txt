# set minimum CMake version
cmake_minimum_required(VERSION 3.8...3.14)

# set cmake policy
if(${CMAKE_VERSION} VERSION_LESS 3.14)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()


# set up top-level project
project(slam_demo
  VERSION 0.0.1
  DESCRIPTION "SLAM demo"
  LANGUAGES CXX)


# set some variables
# compile options
set(DEBUG_OPTIONS "-Wall" "-Wextra" "-Werror" "-pedantic")
# custom OpenCV version
#set(OpenCV_VER 3.4.5)


# declare dependencies (must be called after project() is called!!!)
find_package(OpenCV 4.0 REQUIRED)
# custom OpenCV build: path should contain OpenCVConfig.cmake file!
#find_package(OpenCV ${OpenCV_VER} REQUIRED
#  PATHS "~/Workspace/c++/opencv/opencv-${OpenCV_VER}/build")
find_package(Doxygen QUIET
  REQUIRED COMPONENTS dot)


# add sub directories
add_subdirectory(src)
add_subdirectory(examples)


# add document (basic version)
if(${DOXYGEN_FOUND})
  # 1) set doxygen-related variables
  #    - general options
  set(DOXYGEN_FILE_PATTERNS "*.c" "*.cpp" "*.h" "*.hpp")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_EXTRACT_STATIC YES) # documenting static funcs
  set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/doc)
  #    - math-related options
  set(DOXYGEN_USE_MATHJAX YES)
  set(DOXYGEN_MATHJAX_EXTENSIONS "TeX/AMSmath" "TeX/AMSsymbols")
  #set(DOXYGEN_EXTRA_PACKAGES amsmath)
  # 2) call cmake function to add doc
  doxygen_add_docs(doc
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/examples
    )
endif()
