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
#include "MapPoint.hpp"

namespace SLAM_demo {

// forward declarations
class Frame;
class Map;

/**
 * @class Tracker
 * @brief Track adjacent 2 frames for feature extraction and matching.
 */
class Tracker {
public: // public data
    /// A threshold in Lowe's ratio test for discarding wrong matches.
    static const float TH_DIST;
    /// Max ratio of reprojection error of F to that of H.
    static const float TH_MAX_RATIO_FH;
    /// Cosine of smallest appropriate parallax/angle between 2 views.
    static const float TH_COS_PARALLAX;
    /// Reprojection error factor for checking good triangulated points.
    static const float TH_REPROJ_ERR_FACTOR;
    /// For selecting best possible recovered pose.
    static const float TH_POSE_SEL;
    /// Minimum ratio of triangulated points to total keypoint matches,
    static const float TH_MIN_RATIO_TRIANG_PTS;
    /// Minimum number of 3D-to-2D matches for pose estimation using PnP.
    static const int TH_MIN_MATCHES_3D_TO_2D;
    /// The 1st frame in the initialized SLAM system.
    static unsigned n1stFrame;
public: // public members
    /**
     * @brief Constructor.
     * @param[in] eMode See System::Mode for more information.
     * @param[in] pMap  Pointer to the map.
     */
    Tracker(System::Mode eMode, const std::shared_ptr<Map>& pMap);
    /** 
     * @brief Track an input image in System::Mode::MONOCULAR mode.
     * @param[in] img       Input image.
     * @param[in] timestamp Timestamp info of the input image.
     * @note The input images will be forced to be converted to grayscale
     *       in System::Mode::MONOCULAR mode.
     */
    void trackImgsMono(const cv::Mat& img, double timestamp);
private: // private data
    /// Tracking state.
    enum class State {
        /// Map is not initialized. Using 2D-2D matches for pose estimation.
        NOT_INITIALIZED, 
        OK, ///< Tracking is successful for a new incoming frame.
        LOST ///< Tracking is unsuccessful.
    };
    /// Selection result of the better transformation from F and H.
    enum class FHResult {F, H, NONE /**< Neither H nor F is appropriate. */};
    /// Reprojection error computation scheme.
    enum class ReprojErrScheme {
        F, ///< Fundamental matrix as reprojection transformation.
        H  ///< Homography as reprojection transformation.
    };
    System::Mode meMode; ///< SLAM system mode.
    State meState; ///< Tracking state.
    bool mbFirstFrame; ///< Whether it is the 1st input frame to be processed.
    std::shared_ptr<Map> mpMap; ///< Pointer to the map.
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
    /// Feature matcher.
    std::shared_ptr<cv::DescriptorMatcher> mpFeatMatcher;
    /** 
     * @brief 2D-to-2D matches: match info between 2D keypoints of views 1 
     *        (querying set) & 2 (training set).
     */
    std::vector<cv::DMatch> mvMatches2Dto2D;
    /**
     * @brief 3D-to-2D matches: match info between 3D points of the map 
     *        (querying set) and 2D keypoint of view 2 (training set).
     */
    std::vector<cv::DMatch> mvMatches3Dto2D;
    /**
     * @brief 3D-to-3D matches: match info between 3D points of the map
     *        (querying set) and those triangulated from view 1 & 2 
     *        (training set).
     */
    std::vector<cv::DMatch> mvMatches3Dto3D;
    /**
     * @brief Pointer to all map points. Update once when doing tracking
     *        after the tracking state is Tracker::State::OK.
     */
    std::vector<std::shared_ptr<MapPoint>> mvpMPts;
    /** 
     * @brief Camera poses \f$T_{cw,k|x}\f$ from \f$x\f$th frame to \f$k\f$th 
     *        frame where x = Tracker::n1stFrame.
     */
    std::vector<CamPose> mvTs;
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
     * @brief Match features between previous frame (1) and current frame (2)
     *        for 2D-to-2D case when map has not been initialized.
     * @return A vector of matching keypoints of cv::DMatch type.
     * @note After the feature matching scheme, where candidate keypoint matches
     *       are filterd out using Lowe's ratio test, the candidates whose 
     *       keypoints from both frames are out-of-border after undistorting 
     *       the captured images are discarded.
     */
    void matchFeatures2Dto2D();
    /** 
     * @brief Match features between the map and current frame (2)
     *        for 3D-to-2D case after map is initialized.
     * @return A vector of matching keypoints of cv::DMatch type.
     * @note After the feature matching scheme, where candidate keypoint matches
     *       are filterd out using Lowe's ratio test, the candidates whose 
     *       keypoints (reprojected) from both frames are out-of-border 
     *       after undistorting the captured images are discarded.
     */
    void matchFeatures3Dto2D();
    /** 
     * @brief Check whether a \f$2 \times 1\f$ cv::Mat point is inside 
     *        the border of an undistorted image.
     * @param[in] pt \f$2 \times 1\f$ point of cv::Mat type.
     * @return True if the 2D point is inside the border.
     */
    bool is2DPtInBorder(const cv::Mat& pt) const;
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
     * @brief Compute fundamental matrix F from previous frame to current frame.
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
     * @param[in,out] Xw3Ds       \f$3 \times N\f$ matrix of \f$N\f$ triangulated 
     *                            \f$3 \times 1\f$ world coordinates.
     * @param[in,out] vIdxGoodPts A vector of indices of valid 
     *                            triangulated points. The index is the index of
     *                            matched pair of 2 keypoints of 2 views stored 
     *                            in @p mvMatches.
     * @return True if pose recovery is successful.
     */
    bool recoverPoseFromFH(const cv::Mat& F21, const cv::Mat& H21,
                           cv::Mat& Xw3Ds, std::vector<int>& vIdxGoodPts);
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
    /**
     * @brief Compute reprojection error based on transformation matrix
     *        @p T21 (reproject @p p1 from view 1 to view 2) and @p T12
     *        (reproject @p p2 from view 2 to view 1).
     *
     * Given 2D-to-2D match \f$p_1 = (x_1, y_1, 1)\f$, 
     * \f$p_2 = (x_2, y_2, 1)\f$, and transformation matrix \f$T_{21}\f$,
     * \f$T_{12}\f$,
     * 
     * Reprojection error \f$e_F\f$ for \f$F\f$ where \f$T_{21} = F_{21}\f$ and
     * \f$T_{12} = F_{12} = F_{21}^T\f$:
     * \f{align}{
     *   e_F &= d(p_2, F_{21} p_1)^2 + d(p_1, F_{12} p_2)^2 \\
     *       &= d(p_2, l_2)^2 + d(p_1, l_1)^2 \\
     *       &= \frac{a_2 x_2 + b_2 y_2 + c_2}{a_2^2 + b_2^2} + 
     *          \frac{a_1 x_1 + b_1 y_1 + c_1}{a_1^2 + b_1^2}
     * \f}
     * where \f$l_1 = (a_1, b_1, c_1)\f$ and \f$l_2 = (a_2, b_2, c_2)\f$ are
     * the epipolar lines of triangulated point \f$P\f$ based on \f$p_1\f$, 
     * \f$p_2\f$ in view 1 and 2, respectively.
     *
     * Reprojection error \f$e_H\f$ for \f$H\f$ where \f$T_{21} = H_{21}\f$ and
     * \f$T_{12} = H_{12} = H_{21}^{-1}\f$:
     * \f{align}{
     *   e_H &= d(p_2, H_{21} p_1)^2 + d(p_1, H_{12} p_2)^2 \\
     *       &= d(p_2, p_2')^2 + d(p_1, p_1')^2 \\
     *       &= |x_2 - x_2'|^2 + |y_2 - y_2'|^2 + 
     *          |x_1 - x_1'|^2 + |y_1 - y_1'|^2
     * \f}
     * where \f$p_1' = (x_1', y_1')\f$, \f$p_2' = (x_2', y_2')\f$.
     *
     * @param[in] T21     Transformation matrix from view 1 to view 2.
     * @param[in] T12     Transformation matrix from view 2 to view 1.
     * @param[in] p1      Point in view 1.
     * @param[in] p2      Point in view 2.
     * @param[in] eScheme Computation scheme based on different transformation
     *                    matrices (see Tracker::ReprojErrScheme for details).
     * @return Reprojection error (square form) in FP32 precision.
     */
    float computeReprojErr(const cv::Mat& T21, const cv::Mat& T12,
                           const cv::Mat& p1, const cv::Mat& p2,
                           ReprojErrScheme eScheme) const;
    /**
     * @brief Check whether a triangulated 3D world point is good enough
     *        to be included to the map.
     *
     * Basically there're 4 things to check:
     * - The depth of the 3D camera coordinates in both views (whether the depth
     *   is negative)
     * - The parallax of the 2 views (the angle oP-Xw-oC where oP and oC are 
     *   the camera origins in previous and current frame, Xw is the 
     *   triangulated point) (whether the parallax is too low),
     * - The position of reprojected 2D image point (whether it is outside the 
     *   image border);
     * - The reprojection errors in both views (whether it is too high).
     *
     * @param[in] Xw    \f$3 \times 1\f$ triangulated 3D world point in
     *                  inhomogeneous coordinate.
     * @param[in] kpt1  2D keypoint in previous frame.
     * @param[in] kpt2  corresponding 2D keypoint in current frame.
     * @param[in] pose2 The recovered pose for view 2.
     * @return True if the triangulated point is good enough, otherwise false.
     */
    bool checkTriangulatedPt(const cv::Mat& Xw,
                             const cv::KeyPoint& kpt1, const cv::KeyPoint& kpt2,
                             const CamPose& pose2) const;
    ///@}    
    /// Get camera pose of a target frame relative to 1st frame.
    CamPose getAbsPose(int nIdx) const { return mvTs[nIdx]; }
    /// Set camera pose of current frame relative to 1st frame.
    void setAbsPose(const CamPose& pose) { mvTs.push_back(pose); }
    /**
     * @brief Build initial map from triangulated \f$3 \times 1\f$ world 
     *        coordinates.
     * @param[in] Xws     \f$3 \times N\f$ matrix of \f$N\f$ triangulated 
     *                    \f$3 \times 1\f$ world coordinates.
     * @param[in] vIdxPts A vector of indices of valid triangulated points. 
     *                    The index is the index of matched pair of 2 keypoints
     *                    of 2 views stored in @p Tracker::mvMatches2Dto2D.
     */
    void buildInitialMap(const cv::Mat& Xws,
                         const std::vector<int>& vIdxPts) const;
    /**
     * @brief Track subsuquent frames after the map is initialized.
     * @return Tracking state Tracker::State.
     */
    Tracker::State track();
    /** 
     * @brief Pose estimation after map is initialized. Currently use PnP
     *        based on 3D-to-2D matches.
     */
    Tracker::State poseEstimation();
    /// Triangulate new map points and update the map after pose estimation.
    void updateMap();
    /// Update visibility counter of all existed map points.
    void updateMPtCntObs() const;
    /** 
     * @brief Triangulate new 3D points based on estimated pose and 
     *        2D-to-2D matches.
     * @param[in,out] Xws     \f$3 \times N\f$ matrix of \f$N\f$ triangulated 
     *                        \f$3 \times 1\f$ world coordinates.
     * @param[in,out] vIdxPts A vector of indices of valid triangulated points.
     *                        The index is the index of matched pair of 2 
     *                        keypoints of 2 views stored in
     *                        @p Tracker::mvMatches2Dto2D.
     */
    void triangulate3DPts(cv::Mat& Xws, std::vector<int>& vIdxPts) const;
    /**
     * @brief Add new map points and fuse existed map points based on newly
     *        triangulated 3D points.
     * @param[in] Xws     \f$3 \times N\f$ matrix of \f$N\f$ triangulated 
     *                    \f$3 \times 1\f$ world coordinates.
     * @param[in] vIdxPts A vector of indices of valid triangulated points. 
     *                    The index is the index of matched pair of 2 keypoints
     *                    of 2 views stored in @p Tracker::mvMatches2Dto2D.
     */
    void fuseMPts(const cv::Mat& Xws, const std::vector<int>& vIdxPts) const;
};

} // namespace SLAM_demo

#endif // TRACKER_HPP
