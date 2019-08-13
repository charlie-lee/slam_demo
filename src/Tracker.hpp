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
private: // private member functions
    /** 
     * @brief Match features between current frame (1) and reference frame (2).
     * @param[in]  pFrame1  Pointer to current frame.
     * @param[in]  pFrame2  Pointer to reference frame.
     * @param[in]  TH_DIST  A threshold in Lowe's ratio test for discarding 
     *                      wrong matches (default value: 0.7).
     * @param[out] vMatches A vector of matching keypoints of cv::DMatch type.
     */
    void matchFeatures(std::shared_ptr<Frame> pFrame1,
                       std::shared_ptr<Frame> pFrame2,
                       std::vector<cv::DMatch>& vMatches,
                       const float TH_DIST = 0.7f);
private: // private data
    System::Mode meMode; ///< SLAM system mode.
    cv::Mat mPrevImg; ///< Image of previous frame.
    bool mbFirstFrame; ///< Whether it is the 1st input frame to be processed.
    /// A pointer to current frame (frame 1) for a vector of views.
    std::vector<std::shared_ptr<Frame>> mvpFrames;
    /// A pointer to reference (previous) frame (frame 2) for a vector of views.
    std::vector<std::shared_ptr<Frame>> mvpRefFrames;
    /// Feature matcher.
    std::shared_ptr<cv::DescriptorMatcher> mpFeatMatcher;
};

} // namespace SLAM_demo

#endif // TRACKER_HPP
