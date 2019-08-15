# Changelog & TODO-list

## TODO
- Compute pose {R, t} from 2D-2D keypoint matches (for map initialization)
  - Compute fundamental matrix F (done)
  - Compute homography H (done)
  - Select better transformation from F and H (0%)
  - Decompose {R, t} from F or H (TODO)
- Add a class for storing pose information

## Latest Version
- Fix some coding style issues
- Improve performance on feature matching based on implementations on OpenCV
- Feature matching on undistorted keypoint data
- Display feature matching result on undistorted images
- Update class interface design
- Compute pose {R, t} from 2D-2D keypoint matches (for map initialization)
  - Compute fundamental matrix F & homography H
    - Get input data: store these data vs. conversion on the fly
      - Conversion: need to store 2D-2D matches to retrieve the keypoints

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
