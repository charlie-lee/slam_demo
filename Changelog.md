# Changelog & TODO-list

## TODO
- Low priority TODOs
  - Add trajectory visualization mode (TODO)
  - Check why mask info not utilized by knnMatch() for feature matching (TODO)
- Review orientation check procedure used in feature matching schemes

## Latest Version

## v0.6.1
- Some minor update on class designs for more robust code
- Remove obsolete code on keyframe trajectory recording
- Bug fixes:
  - Fix a bug that, when removing a keyframe, the connections of its related 
    keyframes are not correctly updated
    
## v0.6.0
- FeatureMatcher: 
  - Add ROI-based feature matching scheme: for each feature in one frame,
    find a minimum set of features in another frame to match
    - Use Lowe's ratio test
    - Used in tracker and local mapper (both 2D-to-2D and 2D-to-3D cases)
- LocalMapper:
  - Fuse new triangulated points with bound map points based on parallax data
  - Only remove KF data that is in the map (temp fix for redundant KF removal)
- Improve system performance:
  - Fix some tracking losses when encountering large motion by using ROI-based 
    feature matching scheme with a large Hamming distance threshold 
    (64 currently) (contribute most to performance boost)
  - Increase the baseline for a valid map point: higher tracked-to-visible
    ratio (0.6 currently)
  - Create new keyframe more frequently: as long as there're less than
    250 2D-to-3D matches in the tracker for each frame
  - Try to detect and suppress pose outlier for each frame by thresholding on 
    number of inliers before record the pose of current frame

## v0.5.1
- Disable scene depth scaling scheme
- Remove redendant counter for map points
- Add local BA scheme
- Add keyframe trajectory dump (temp version)

## v0.5.0
- Simplify some class designs
- Add FrameBase class and regard Frame as derived class
- Add KeyFrame class as derived class of FrameBase
  - Add weighted graph related member functions
  - Store a map of (pKF, weight (common map points)) pairs
  - Use {idxKpt, pMPt} pair instead of vMPts[idxKpt] data for Frame & KeyFrame 
    classes
- Revamp data structure for matched 3D map points in Frame/KeyFrame class
- Extract feature matching operations to a separate class
- Add LocalMapper class with the following features:
  - New map point creation scheme
  - Map point fusion scheme
  - Keyframe removal scheme
- Scale scene median depth to 1 after initialization
- Add keyframe addition scheme (basic version)
- Update interface for related classes

## v0.4.4
- Do not add frame observation data to the map if there's no new 
  triangulated map points in the frame
- Add outlier detection to single-frame BA scheme
- Fix trajectory dump bug: dump `[R^T | -R^T*t]` instead of `[R | t]`

## v0.4.3
- Use constant velocity model for initial pose guess when tracking
- Force skip BA optimization if there're not enough map point nodes in the graph
- Bug fixes:
  - Fix a bug about the copy behavior of CamPose objects, which causes
    the failure of epipolar constraint test when triangulating new map points
  - Fix wrong camera intrinsic data for Freiburg1 sequences in TUM dataset

## v0.4.2
- Revise tracking procedure
- Add outlier removal for global BA (currently disabled)
- Modify single-frame BA to have multiple pose nodes in the optimization graph
  (all but the pose of current frame are fixed)

## v0.4.1
- Force update map point descriptor to be the latest extracted one
- Update single-frame BA scheme: optimize all observed map points instead of
  only newly triangulated ones
- Add support for windowed BA for global BA function
- Reset trajectory if the system is re-initialized
- Bug fixes:
  - Fix observe-to-visible ratio computation
  - Fix ROI computation for block feature extraction scheme
  - Fix memory leak for Frame objects after outlier map points are removed

## v0.4.0
- Integrate g2o library into the project
- Revamp matching result display scheme for more intuitive representation
- Add global bundle adjustment scheme (seems broken!)
- Add pose optimization scheme
- Add single-frame bundle adjustment scheme
- Update pose information in Tracker after BA is done for final 
  trajectory dump
- Restore map point descriptor update scheme for map point fusion scheme

## v0.3.1
- Update design of MapPoint & Frame class for bundle adjustment support
  - Update MapPoint class
    - Update constructor interface
    - Implement addObservation({pFrame, nIdxKpt})
    - Implement getKpt({pFrame, nIdxKpt})
    - Implement getDesc({pFrame, nIdxKpt})
  - Update Frame class
    - Implement getObservedMapPoints() (return a set of map points)
- Update feature extraction scheme: try to find at least 1 keypoint for each
  image block

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
