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

namespace SLAM_demo {

// forward declarations
class Frame;

/**
 * @class Tracker
 * @brief Track adjacent 2 frames for feature extraction and matching.
 */
class Tracker {
public:
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
    /// Selection result of the better transformation from F and H.
    enum class FHResult {F, H, NONE /**< Neither H nor F is appropriate. */};
    System::Mode meMode; ///< SLAM system mode.
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
private: // private member functions
    /// Convert input image @p img into grayscale image.
    cv::Mat rgb2Gray(const cv::Mat& img) const;
    /// Initialize 3D map of SLAM system.
    bool initializeMap();
    /** 
     * @brief Initialize 3D map of SLAM system for System::Mode::MONOCULAR case.
     * @param[in] pFPrev Pointer to previous frame.
     * @param[in] pFCur  Pointer to current frame.
     */
    bool initializeMapMono(const std::shared_ptr<Frame>& pFPrev,
                           const std::shared_ptr<Frame>& pFCur);
    /** 
     * @brief Match features between previous frame (1) and current frame (2).
     * @param[in]  pFPrev  Pointer to previous frame.
     * @param[in]  pFCur   Pointer to current frame.
     * @param[in]  TH_DIST A threshold in Lowe's ratio test for discarding 
     *                     wrong matches (default value: 0.7).
     * @return A vector of matching keypoints of cv::DMatch type.
     * @note After the feature matching scheme, where candidate keypoint matches
     *       are filterd out using Lowe's ratio test, the candidates whose 
     *       keypoints from both frames are out-of-border after undistorting 
     *       the captured images are discarded.
     */
    std::vector<cv::DMatch> matchFeatures2Dto2D(
        const std::shared_ptr<Frame>& pFPrev,
        const std::shared_ptr<Frame>& pFCur,
        const float TH_DIST = 0.7f) const;
    /** 
     * @brief Check whether a cv::KeyPoint keypoint is inside the border 
     *        of an undistorted image.
     * @param[in] kpt Keypoint of type cv::KeyPoint.
     * @return true if the keypoint is inside the border.
     */
    bool isKptInBorder(const cv::KeyPoint& kpt) const;
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
     * @name groupFHComputation
     * @brief A group of functions related to fundamental matrix F & 
     *        homography H computation, and reconstruction of pose 
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
     * @brief Reconstruct pose \f$[R|t]\f$ from either fundamental matrix F
     *        or homography H.
     * @return True if pose reconstruction is successful.
     */
    bool reconstructPoseFromFH(const std::shared_ptr<Frame>& pFPrev,
                               const std::shared_ptr<Frame>& pFCur,
                               const std::vector<cv::DMatch>& vMatches,
                               const cv::Mat& Fcp, const cv::Mat& Hcp) const;
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
     * @brief Reconstruct pose \f$[R|t]\f$ from fundamental matrix F.
     * @return True if pose reconstruction based on F is successful.
     */
    bool reconstructPoseFromF(const std::shared_ptr<Frame>& pFPrev,
                              const std::shared_ptr<Frame>& pFCur,
                              const std::vector<cv::DMatch>& vMatches,
                              const cv::Mat& Fcp) const;
    /**
     * @brief Reconstruct pose \f$[R|t]\f$ from homography H.
     * @return True if pose reconstruction based on H is successful.
     */
    bool reconstructPoseFromH(const std::shared_ptr<Frame>& pFPrev,
                              const std::shared_ptr<Frame>& pFCur,
                              const std::vector<cv::DMatch>& vMatches,
                              const cv::Mat& Hcp) const;
    ///@} // end of groupFHComputation
};

} // namespace SLAM_demo

#endif // TRACKER_HPP
