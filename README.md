# SLAM Demo

A Simultaneous Localization and Mapping (SLAM) demo application.

## Dependencies
- [OpenCV](https://opencv.org/)

## Build Programs & Documentations
1. Build CMake Files
   - `cd build && cmake ..`
2. Build Programs
   - `make`
3. Build Documentations (optional)
   - `make doc`

## Build Tags
1. Create symbolic links to included libraries
   - `cd /path/to/slam_demo`
   - `mkdir tags`
   - `ln -s /usr/include/libX /usr/include/libY ... tags`
2. Create **ctags** and **gtags**
   - `cd /path/to/slam_demo`
   - `ctags -e -R --c++-kinds=+px --fields=+iaS --extra=+q .`
   - `gtags`

## Run Examples
1. mono_vo
   - `mono_vo path/to/cfgFile.yaml path/to/imgPath path/to/camDataFile.csv`
