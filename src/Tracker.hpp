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

/**
 * @class Tracker
 * @brief Track adjacent 2 frames for feature extraction and matching.
 */
class Tracker {
public: // public data
    /// A threshold in Lowe's ratio test for discarding wrong matches.
    static const float TH_DIST;
    /** 
     * @brief Similarity threshold between reprojection error of F and H when
     *        selecting the better representation of the transformation from
     *        previous frame to current frame.
     */
    static const float TH_SIMILARITY;
    /// Cosine of smallest appropriate parallax/angle between 2 views.
    static const float TH_COS_PARALLAX;
    /// For selecting best possible recovered pose.
    static const float TH_POSE_SEL;
    /// Minimum ratio of triangulated points to total keypoint matches,
    static const float TH_MIN_RATIO_TRIANG_PTS;
public: // public members
    /**
     * @brief Constructor.
     * @param[in] eMode See System::Mode for more information.
     */
    Tracker(System::Mode eMode);
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
    State mState; ///< Tracking state.
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
    bool mbFirstFrame; ///< Whether it is the 1st input frame to be processed.
    /// A pointer to previous frame (frame 1) for a vector of views.
    std::vector<std::shared_ptr<Frame>> mvpFramesPrev;
    /// A pointer to current frame (frame 2) for a vector of views.
    std::vector<std::shared_ptr<Frame>> mvpFramesCur;
    /// Feature matcher.
    std::shared_ptr<cv::DescriptorMatcher> mpFeatMatcher;
    /// Camera poses \f$T_{cw,k|1}\f$ w.r.t. 1st frame.
    std::vector<CamPose> mvTs;
private: // private member functions
    /// Convert input image @p img into grayscale image.
    cv::Mat rgb2Gray(const cv::Mat& img) const;
    /// Initialize 3D map of SLAM system.
    State initializeMap();
    /** 
     * @brief Initialize 3D map of SLAM system for System::Mode::MONOCULAR case.
     * @param[in] pFPrev Pointer to previous frame.
     * @param[in] pFCur  Pointer to current frame.
     * @return State::OK if the initialization is successful, otherwise
     *         State::NOT_INITIALIZED.
     */
    State initializeMapMono(const std::shared_ptr<Frame>& pFPrev,
                            const std::shared_ptr<Frame>& pFCur);
    /** 
     * @brief Match features between previous frame (1) and current frame (2).
     * @param[in]  pFPrev  Pointer to previous frame.
     * @param[in]  pFCur   Pointer to current frame.
     * @return A vector of matching keypoints of cv::DMatch type.
     * @note After the feature matching scheme, where candidate keypoint matches
     *       are filterd out using Lowe's ratio test, the candidates whose 
     *       keypoints from both frames are out-of-border after undistorting 
     *       the captured images are discarded.
     */
    std::vector<cv::DMatch> matchFeatures2Dto2D(
        const std::shared_ptr<Frame>& pFPrev,
        const std::shared_ptr<Frame>& pFCur) const;
    /** 
     * @brief Check whether a \f$2 \times 1\f$ cv::Mat point is inside 
     *        the border of an undistorted image.
     * @param[in] pt \f$2 \times 1\f$ point of cv::Mat type
     * @return True if the 2D point is inside the border.
     */
    bool is2DPtInBorder(const cv::Mat& pt) const;
    /** 
     * @brief Display feature matching results of adjacent 2 **UNDISTORTED**
     *        frames (left for current/left frame and right for 
     *        reference/previous/right frame).
     * @param[in] vMatches Keypoint matches of current and reference frame.
     * @param[in] viewPrev View index of previous frame.
     * @param[in] viewCur  View index of current frame.
     */
    void displayFeatMatchResult(const std::vector<cv::DMatch>& vMatches,
                                int viewPrev = 0, int viewCur = 0) const;
    /**
     * @name F/H Computation Functions
     * @brief A group of functions related to fundamental matrix F & 
     *        homography H computation, and recovery of pose 
     *        \f$[R|t]\f$ using F & H.
     * @param[in] pFPrev   Pointer to previous frame.
     * @param[in] pFCur    Pointer to current frame.
     * @param[in] vMatches A vector 2D-to-2D keypoint matches.
     * @param[in] Fcp      Fundamental matrix F from previous to current frame.
     * @param[in] Hcp      Homography H from previous to current frame.
     */
    ///@{
    /** 
     * @brief Compute fundamental matrix F from previous frame to current frame.
     * @return Fundamental matrix F, or an empty matrix if computation failed.
     */
    cv::Mat computeFundamental(const std::shared_ptr<Frame>& pFPrev,
                               const std::shared_ptr<Frame>& pFCur,
                               const std::vector<cv::DMatch>& vMatches) const;
    /**
     * @brief Compute homography H from previous frame to current frame.
     * @return Homography H, or an empty matrix if computation failed.
     */
    cv::Mat computeHomography(const std::shared_ptr<Frame>& pFPrev,
                              const std::shared_ptr<Frame>& pFCur,
                              const std::vector<cv::DMatch>& vMatches) const;
    /**
     * @brief Recover pose \f$[R|t]\f$ from either fundamental matrix F
     *        or homography H.
     * @return True if pose recovery is successful.
     */
    bool recoverPoseFromFH(const std::shared_ptr<Frame>& pFPrev,
                           const std::shared_ptr<Frame>& pFCur,
                           const std::vector<cv::DMatch>& vMatches,
                           const cv::Mat& Fcp, const cv::Mat& Hcp);
    /**
     * @brief Select the better transformation of 2D-to-2D point matches
     *        from fundamental matrix F and homography H.
     * @return Tracker::FHResult result.
     */
    FHResult selectFH(const std::shared_ptr<Frame>& pFPrev,
                      const std::shared_ptr<Frame>& pFCur,
                      const std::vector<cv::DMatch>& vMatches,
                      const cv::Mat& Fcp, const cv::Mat& Hcp) const;
    /**
     * @brief Decompose fundamental matrix \f$F\f$ to get possible pose
     *        \f$[R|t]\f$ solutions. Camera intrinsics \f$K\f$ is needed.
     * @param[in]     Fcp   Fundamental matrix to be decomposed. 
     * @param[in,out] vRcps A vector of possible rotation matrices (
     *                      with \f$i\f$th pose being {vRcps[i], vtcps[i]}).
     * @param[in,out] vtcps A vector of possible translation matrices (
     *                      with \f$i\f$th pose being {vRcps[i], vtcps[i]}).
     * @return Number of possible pose solutions.
     */
    int decomposeFforRT(const cv::Mat& Fcp,
                        std::vector<cv::Mat>& vRcps,
                        std::vector<cv::Mat>& vtcps) const;
    /**
     * @brief Decompose homography \f$H\f$ to get possible pose
     *        \f$[R|t]\f$ solutions.
     * @param[in]     Hcp        Homography to be decomposed. 
     * @param[in,out] vRcps      A vector of possible rotation matrices (with
     *                           \f$i\f$th pose being {vRcps[i], vtcps[i]}).
     * @param[in,out] vtcps      A vector of possible translation matrices (with
     *                           \f$i\f$th pose being {vRcps[i], vtcps[i]}).
     * @param[in,out] vNormalcps A vector of possible normal planes 
     *                           (vNormalcps[i] for \f$i\f$th pose).
     * @return Number of possible pose solutions.
     */
    int decomposeHforRT(const cv::Mat& Hcp,
                        std::vector<cv::Mat>& vRcps,
                        std::vector<cv::Mat>& vtcps,
                        std::vector<cv::Mat>& vNormalcps) const;
    /**
     * @brief Compute reprojection error based on transformation matrix
     *        @p T21 (reproject @p p1 from view 1 to view 2) and @p T12
     *        (reproject @p p2 from view 2 to view 1).
     * @param[in] T21 Transformation matrix from view 1 to view 2.
     * @param[in] T12 Transformation matrix from view 2 to view 1.
     * @param[in] p1  Point in view 1.
     * @param[in] p2  Point in view 2.
     * @param[in] scheme Computation scheme based on different transformation
     *                   matrices (see Tracker::ReprojErrScheme for details).
     * @return Reprojection error (square form) in FP32 precision.
     */
    float computeReprojErr(const cv::Mat& T21, const cv::Mat& T12,
                           const cv::Mat& p1, const cv::Mat& p2,
                           ReprojErrScheme scheme) const;
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
     * @param[in] Xw   \f$4 \times 1\f$ triangulated 3D world point in
     *                 homogeneous coordinate.
     * @param[in] kptP 2D keypoint in previous frame.
     * @param[in] kptC corresponding 2D keypoint in current frame.
     * @param[in] pose The recovered pose.
     * @return True if the triangulated point is good enough, otherwise false.
     */
    bool checkTriangulatedPt(const cv::Mat& Xw,
                             const cv::KeyPoint& kptP, const cv::KeyPoint& kptC,
                             const CamPose& pose) const;
    ///@}    
    /// Get camera pose of a target frame relative to 1st frame.
    CamPose getAbsPose(unsigned nIdx) const { return mvTs[nIdx]; }
    /// Set camera pose of current frame relative to 1st frame.
    void setAbsPose(const CamPose& pose) { mvTs.push_back(pose); }
};

} // namespace SLAM_demo

#endif // TRACKER_HPP
