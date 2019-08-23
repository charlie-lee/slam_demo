/**
 * @file   Frame.hpp
 * @brief  Header of Frame class for storing intra- and inter-frame info.
 * @author Charlie Li
 * @date   2019.08.12
 */

#ifndef FRAME_HPP
#define FRAME_HPP

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "CamPose.hpp"

namespace SLAM_demo {

/**
 * @class Frame
 * @brief Store intra- and inter-frame info, including keypoint data, 
 *        camera pose, etc.
 */
class Frame {
public: // public data
    CamPose mPose; ///< Camera pose of the current frame.
public: // public members
    /**
     * @brief Constructor of the Frame class.
     *
     * Features in the input image will be extracted by a feature extractor
     * into keypoint and descriptor data.
     *
     * @param[in] img       Input image.
     * @param[in] timestamp Timestamp info of current frame.
     */
    Frame(const cv::Mat& img, double timestamp);
    /// Get keypoint data for current frame.
    std::vector<cv::KeyPoint> getKeyPoints() const { return mvKpts; }
    /// Get feature descriptor for current frame.
    cv::Mat getFeatDescriptors() const { return mDescs; }
private: // private data
    double mTimestamp; ///< Timestamp info for the current frame.
    std::shared_ptr<cv::Feature2D> mpFeatExtractor; ///< Feature extractor.
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    cv::Mat mDescs; ///< Descriptor of the current frame.
private: // private member functions
    /// Extract features from the current frame.
    void extractFeatures(const cv::Mat& img);
    /**
     * @brief Undistort keypoint coordinates based on camera intrinsics and
     *        distortion coefficients.
     */
    void undistortKpts();
};

} // namespace SLAM_demo

#endif // FRAME_HPP