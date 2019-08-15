# Changelog & TODO-list

## TODO
- Compute pose {R, t} from 2D-2D keypoint matches
  - Compute fundamental matrix F
  - Compute homography H
  - Select better transformation from F and H
  - Decompose {R, t} from F or H
- Add a class for storing pose information

## Latest Version
- Fix some coding style issues
- Improve performance on feature matching based on implementations on OpenCV
- Feature matching on undistorted keypoint data
- Display feature matching result on undistorted images

## v0.0.3
- Add system initialzation scheme
  - Add basic structure of Tracker class
  - Add Frame class for feature extraction & matching scheme
    - Currently only implementations from OpenCV are used
- Add system parameter loading scheme
  - Write system parameter YAML file
  - Create a singleton Config class for global cfgs

## v0.0.2
- Add image data loading scheme for monocular visual odometry example

## v0.0.1
- Set up basic project structure
