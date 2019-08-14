# Changelog & TODO-list

## TODO
- Compute camera pose {R, t} from 2D-2D keypoint matches

## Latest Version
- Fix some coding style issues
- Improve performance on feature matching based on implementations on OpenCV

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
