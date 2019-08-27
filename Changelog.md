# Changelog & TODO-list

## TODO
- Compute pose [R|t] from 2D-2D keypoint matches (for map initialization)
  - Store triangulated 3D world points to map (TODO)
- Add Map and MapPoint class for storing map data (TODO)

## Latest Version
- Add pose display function
- Fix reprojection error computation for H when selecting pose recovery model

## v0.0.4
- Fix some coding style issues
- Improve performance on feature matching based on implementations on OpenCV
- Feature matching on undistorted keypoint data
- Display feature matching result on undistorted images
- Update class interface design
- Compute pose {R, t} from 2D-2D keypoint matches (for map initialization)
  - Compute fundamental matrix F & homography H
  - Select better transformation from F and H
    - Fix computation of reprojection error of F
  - Recover pose [R|t] from H or F
- Add CamPose class for storing pose information
  - Basic constructors & getters/setters
  - Getters for other pose representations

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
