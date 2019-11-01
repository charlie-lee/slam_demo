/**
 * @file   Tracker.hpp
 * @brief  Header of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#ifndef TRACKER_HPP
#define TRACKER_HPP

#include "System.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class Frame;
class KeyFrame;
class LocalMapper;
class Map;
class MapPoint;
class Optimizer;

/**
 * @class Tracker
 * @brief Track adjacent 2 frames for feature extraction and matching.
 *
 * The procedures in tracker class are as follows:
 *
 * For each incoming frame, do feature extraction, and then,
 * 1. Do map initialization if the tracking state is being not initialized;
 *    - Do 2D-to-2D feature matching between the latest 2 views;
 *    - Compute fundamental matrix and homography between the 2 views;
 *    - Recover initial pose for the 2nd view from the 2 transformations
 *      computed in the previous step;
 *    - If the pose recovery is successful, build initial map;
 *    - Set tracking state to OK for the core tracking procedure.
 * 2. Start tracking incoming frames after the tracking state becomes OK;
 *    - Do 2D-to-3D feature matching between the keypoints in the latest view
 *      and the tracked map points in the previous view;
 *    - Estimate absolute pose using PnP given 2D-to-3D matches;
 *      - Use pose of previous view as the initial guess;
 *    - Do pose optimization on the above estimated pose;
 *    - Track local map (connected keyframes with the last added keyframe,
 *      and the keyframes that observe the matched map points in the previous 
 *      view) for more 2D-to-3D matches;
 *    - Redo 2D-to-3D feature matching between the latest view and all the 
 *      visible map points;
 *    - Do pose optimization on the previously optimized pose;
 *    - Check whether current frame is ready to be transformed to a keyframe
 *    - If the current frame is quelified, transform it to a keyframe
 *      and add it to the local mapper.
 * 3. Return to 1 and continue track next incoming frame:
 *    - If the tracking failed consecutively for some frames, re-initialize
 *      the system.
 */
class Tracker {
public: // public data
    /// Tracking state.
    enum class State {
        /// Map is not initialized. Using 2D-2D matches for pose estimation.
        NOT_INITIALIZED, 
        OK, ///< Tracking is successful for a new incoming frame.
        LOST ///< Tracking is unsuccessful.
    };
    /// The 1st frame in the initialized SLAM system.
    static unsigned n1stFrame;
public: // public members
    /**
     * @brief Constructor.
     * @param[in] eMode        See System::Mode for more information.
     * @param[in] pMap         Pointer to the map.
     * @param[in] pLocalMapper Pointer to the local mapper.
     */
    Tracker(System::Mode eMode, const std::shared_ptr<Map>& pMap,
            const std::shared_ptr<Optimizer>& pOptimizer,
            const std::shared_ptr<LocalMapper>& pLocalMapper);
    /** 
     * @brief Track an input image in System::Mode::MONOCULAR mode.
     * @param[in] img       Input image.
     * @param[in] timestamp Timestamp info of the input image.
     * @note The input images will be forced to be converted to grayscale
     *       in System::Mode::MONOCULAR mode.
     */
    void trackImgsMono(const cv::Mat& img, double timestamp);
    /// Get state of the tracker.
    Tracker::State getState() const { return meState; }
    /// Set state of the tracker.
    void setState(Tracker::State eState) { meState = eState; }
    /// Get the vector index of the last pose recorded.
    int getIdxLastPose() const { return mvTs.size() - 1; }
    /// Get camera pose of a target frame relative to 1st frame.
    CamPose getAbsPose(int nIdx) const { return mvTs[nIdx]; }
private: // private data
    /// Selection result of the better transformation from F and H.
    enum class FHResult {F, H, NONE /**< Neither H nor F is appropriate. */};
    /// Max ratio of reprojection error of F to that of H.
    static const float TH_MAX_RATIO_FH;
    /// Reprojection error factor for RANSAC PnP scheme.
    static const float TH_REPROJ_ERR_FACTOR;
    /// For selecting best possible recovered pose.
    static const float TH_POSE_SEL;
    /// Minimum ratio of triangulated points to total keypoint matches,
    static const float TH_MIN_RATIO_TRIANG_PTS;
    /// Minimum number of 2D-to-3D matches for pose estimation using PnP.
    static const int TH_MIN_MATCHES_2D_TO_3D;
    System::Mode meMode; ///< SLAM system mode.
    State meState; ///< Tracking state.
    bool mbFirstFrame; ///< Whether it is the 1st input frame to be processed.
    std::shared_ptr<Map> mpMap; ///< Pointer to the map.
    std::shared_ptr<Optimizer> mpOpt; ///< Pointer to the optimizer.
    std::shared_ptr<LocalMapper> mpLocalMapper; ///< Pointer to local mapper.
    /** 
     * @brief Images of previous frame for a vector of views. 
     * @note Previous frame (maybe adjacent) for System::Mode::MONOCULAR and 
     *       System::Mode::RGBD case, and left image for 
     *       System::Mode::STEREO case.
     */
    std::vector<cv::Mat> mvImgsPrev; 
    /**
     * @brief Images of current frame for a vector of views.
     * @note Current frame for System::Mode::MONOCULAR and 
     *       System::Mode::RGBD case, and right image for 
     *       System::Mode::STEREO case.
     */
    std::vector<cv::Mat> mvImgsCur;
    /// Pointers to previous frame (frame 1) for a vector of views.
    std::vector<std::shared_ptr<Frame>> mvpFramesPrev;
    /// Pointers to current frame (frame 2) for a vector of views.
    std::vector<std::shared_ptr<Frame>> mvpFramesCur;
    /**
     * @brief Pointer to the frame of 1st view for feature matching.
     * @note The selected view is as follows:
     *       - Previous frane for System::Mode::MONOCULAR mode;
     *       - Left view of current frame for System::Mode::STEREO mode;
     *       - RGB view of previous frame for System::Mode::RGBD mode.
     */
    std::shared_ptr<Frame> mpView1;
    /**
     * @brief Pointer to the frame of 2nd view for feature matching.
     * @note The selected view is as follows:
     *       - Current frane for System::Mode::MONOCULAR mode;
     *       - Right view of current frame for System::Mode::STEREO mode;
     *       - RGB view of current frame for System::Mode::RGBD mode.
     */
    std::shared_ptr<Frame> mpView2;
    /** 
     * @brief 2D-to-2D matches: match info between 2D keypoints of views 2
     *        (querying set) & 1 (training set).
     */
    std::vector<cv::DMatch> mvMatches2Dto2D;
    /**
     * @brief 2D-to-3D matches: match info between 2D keypoint of view 2 
     *        (querying set) and 3D points of the local map (training set).
     */
    std::vector<cv::DMatch> mvMatches2Dto3D;
    /**
     * @brief 3D-to-3D matches: match info between 3D points of the map
     *        (querying set) and those triangulated from view 2 & 1
     *        (training set).
     */
    std::vector<cv::DMatch> mvMatches3Dto3D;
    /// Pointer to all visible map points.
    std::vector<std::shared_ptr<MapPoint>> mvpMPts;
    /** 
     * @brief Camera poses \f$T_{cw,k|x}\f$ from \f$x\f$th frame to \f$k\f$th 
     *        frame where x = Tracker::n1stFrame.
     */
    std::vector<CamPose> mvTs;
    /// Velocity of current frame/view, i,e, T_{k|k-1}.
    CamPose mVelocity;
    /// Pointer to the latest added keyframe.
    std::shared_ptr<KeyFrame> mpKFLatest;
private: // private member functions
    /// Convert input image @p img into grayscale image.
    cv::Mat rgb2Gray(const cv::Mat& img) const;
    /// Initialize 3D map of SLAM system.
    Tracker::State initializeMap();
    /** 
     * @brief Initialize 3D map of SLAM system for System::Mode::MONOCULAR case.
     * @return State::OK if the initialization is successful, otherwise
     *         State::NOT_INITIALIZED.
     */
    Tracker::State initializeMapMono();
    /** 
     * @brief Display feature matching results of adjacent 2 **UNDISTORTED**
     *        frames (left for current/left frame and right for 
     *        reference/previous/right frame).
     * @param[in] viewPrev View index of previous frame.
     * @param[in] viewCur  View index of current frame.
     */
    void displayFeatMatchResult(int viewPrev = 0, int viewCur = 0) const;
    /**
     * @name F/H Computation Functions
     * @brief A group of functions related to fundamental matrix F & 
     *        homography H computation, and recovery of pose 
     *        \f$[R|t]\f$ using F & H.
     * @param[in] F21      Fundamental matrix F from previous to current frame.
     * @param[in] H21      Homography H from previous (1) to current (2) frame.
     */
    ///@{
    /** 
     * @brief Compute fundamental matrix F from previous frame (view 1) to 
     *        current frame (view 2) based on 2D-to-2D matches.
     * @return Fundamental matrix F, or an empty matrix if computation failed.
     */
    cv::Mat computeFundamental() const;
    /**
     * @brief Compute homography H from previous frame to current frame.
     * @return Homography H, or an empty matrix if computation failed.
     */
    cv::Mat computeHomography() const;
    /**
     * @brief Recover pose \f$[R|t]\f$ from either fundamental matrix F
     *        or homography H.
     * @param[in]     F21
     * @param[in]     H21
     * @param[in,out] Xw3Ds        \f$3 \times N\f$ matrix of \f$N\f$ 
     *                             triangulated \f$3 \times 1\f$ world 
     *                             coordinates.
     * @param[in,out] vnIdxGoodPts A vector of indices of valid triangulated 
     *                             points. The index is the index of matched
     *                             pair of 2 keypoints of 2 views stored in
     *                             @p mvMatches.
     * @return True if pose recovery is successful.
     */
    bool recoverPoseFromFH(const cv::Mat& F21, const cv::Mat& H21,
                           cv::Mat& Xw3Ds, std::vector<int>& vnIdxGoodPts);
    /**
     * @brief Select the better transformation of 2D-to-2D point matches
     *        from fundamental matrix F and homography H.
     * @return Tracker::FHResult result.
     */
    FHResult selectFH(const cv::Mat& F21, const cv::Mat& H21) const;
    /**
     * @brief Decompose fundamental matrix \f$F\f$ to get possible pose
     *        \f$[R|t]\f$ solutions. Camera intrinsics \f$K\f$ is needed.
     * @param[in]     F21   Fundamental matrix to be decomposed. 
     * @param[in,out] vR21s A vector of possible rotation matrices (
     *                      with \f$i\f$th pose being {vR21s[i], vt21s[i]}).
     * @param[in,out] vt21s A vector of possible translation matrices (
     *                      with \f$i\f$th pose being {vR21s[i], vt21s[i]}).
     * @return Number of possible pose solutions.
     */
    int decomposeFforRT(const cv::Mat& F21,
                        std::vector<cv::Mat>& vR21s,
                        std::vector<cv::Mat>& vt21s) const;
    /**
     * @brief Decompose homography \f$H\f$ to get possible pose
     *        \f$[R|t]\f$ solutions.
     * @param[in]     H21        Homography to be decomposed. 
     * @param[in,out] vR21s      A vector of possible rotation matrices (with
     *                           \f$i\f$th pose being {vR21s[i], vt21s[i]}).
     * @param[in,out] vt21s      A vector of possible translation matrices (with
     *                           \f$i\f$th pose being {vR21s[i], vt21s[i]}).
     * @param[in,out] vNormal21s A vector of possible normal planes 
     *                           (vNormal21s[i] for \f$i\f$th pose).
     * @return Number of possible pose solutions.
     */
    int decomposeHforRT(const cv::Mat& H21,
                        std::vector<cv::Mat>& vR21s,
                        std::vector<cv::Mat>& vt21s,
                        std::vector<cv::Mat>& vNormal21s) const;
    ///@}    
    /// Set camera pose of current frame relative to 1st frame.
    void setAbsPose(const CamPose& pose) { mvTs.push_back(pose); }
    /**
     * @brief Build initial map from triangulated \f$3 \times 1\f$ world 
     *        coordinates.
     * @param[in] Xws      \f$3 \times N\f$ matrix of \f$N\f$ triangulated 
     *                     \f$3 \times 1\f$ world coordinates.
     * @param[in] vnIdxPts A vector of indices of valid triangulated points. 
     *                     The index is the index of matched pair of 2 keypoints
     *                     of 2 views stored in @p Tracker::mvMatches2Dto2D.
     */
    void buildInitialMap(const cv::Mat& Xws,
                         const std::vector<int>& vnIdxPts);
    /**
     * @brief Track subsuquent frames after the map is initialized.
     * @return Tracking state Tracker::State.
     */
    Tracker::State track();
    /** 
     * @brief Pose estimation after map is initialized. Currently use PnP
     *        based on 2D-to-3D matches.
     * @param[in] pose Initial guess of the pose for current frame.
     * @return Number of inlier 2D-to-3D matches.
     */
    int poseEstimation(const CamPose& pose);
    /// Track local map for more 2D-to-3D map points matches for current frame.
    std::vector<std::shared_ptr<MapPoint>> trackLocalMap();
    /// Update visibility counter of all existed map points for current view.
    void updateMPtTrackedData() const;
    /// Check whether current frame is qualified as keyframe.
    bool qualifiedAsKeyFrame() const;
    /// Add current frame as new keyframe to the map and local mapper.
    void addNewKeyFrame();
};

} // namespace SLAM_demo

#endif // TRACKER_HPP
