# Changelog & TODO-list

## TODO
- Check how to improve number of consistent matches (same feature matched for 2
  consecutive frames)
- Check how to increase 3D-to-2D matches
- Store last k frames in Tracker class (maybe?)

## Latest version
- Only update map if tracking is successful
- Set looser conditions in good triangulated points checking procedure

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
