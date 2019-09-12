# Changelog & TODO-list

## TODO
- Store last k frames in Tracker class (?)
- Find core issues in tracking procedure
  - Core issue 1: feature matching result not robust enough -> fixed by using
    symmetric test instead of Lowe's ratio test
  - Core issue 2: too few new good triangulated 3D points during tracking
    - Caused by wrong condition (parallax computation) for post-triangulation 
      check -> fixed
- Add essential pose & map optimization scheme (BA) (TODO)
- Skip frames and estimate pose only on keyframes (?)

## Latest version

## v0.3.0
- Visualize 3D-to-2D matching result
- Improve feature matching quality by imposing additional matching criteria
  - Distance between reprojected image point in view 1 and image point in 
    view 2 should be within a threshold
  - Use symmetric test instead of Lowe's ratio test to reject outlier matches
    (Core issue 1)
- Implement block-based feature extraction scheme
- Add fundamental matrix computation based on known pose for post-triangulation
  checking (epipolar constraint does not applied currently)
- Bug fixes
  - **Fix parallax computation for triangulated points (Core issue 2)**

## v0.2.0
- Optimize tracking procedure
  - PnP twice: use previous pose for 1st pass, and use estimated pose 
    for 2nd pass
  - Use newly triangulated map points for 2nd PnP
  - Set outlier flag for map points during pose estimation procedure using RANSAC
  - Map point fusion: only update feature descriptors
- Adjust some system parameters
- Implement trajectory dump for evaluation (currently for TUM dataset only)
- Bug fixes
  - Fix PnP error for undistorting image points for a second time
  - Fix wrong map point counter data (matching consistency)

## v0.1.0
- Update Tracker class design to remove redundant member function parameters
- Implement feature matching scheme for 3D-to-2D case
- Implement basic pose estimation scheme using PnP algorithm
- Update map data after pose of current frame is computed
  - Triangulate 3D points from 2D-to-2D matches
  - Add new 3D points as map points and update data of existing map points
  - Update related counters of all map points 
  - Remove obselete map points
- Fix wrong 3D point being added to the map when building initial map
- Draw tracking results

## v0.0.5
- Add pose display function
- Fix reprojection error computation for H when selecting pose recovery model
- Map initialization: currently traverse all possible pose recovery results 
  from decompsing F & H for a best pose
- Add Map and MapPoint class for storing map data
- Store triangulated 3D world points to build an initial map

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
