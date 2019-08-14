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

namespace SLAM_demo {

/**
 * @class Frame
 * @brief Store intra- and inter-frame info, including keypoint data, 
 *        camera pose, etc.
 */
class Frame {
public:
    /**
     * @brief Constructor of the Frame class.
     *
     * Features in the input image will be extracted by a feature extractor
     * into keypoint and descriptor data.
     *
     * @param[in] img Input image.
     */
    Frame(const cv::Mat& img);
    /// Get keypoint data for current frame.
    std::vector<cv::KeyPoint> getKeyPoints() { return mvKpts; }
    /// Get feature descriptor for current frame.
    cv::Mat getFeatDescriptors() { return mDescs; }
private: // private member functions
    /// Extract features from the current frame.
    void extractFeatures(const cv::Mat& img);
    /**
     * @brief Undistort keypoint coordinates based on camera intrinsics and
     *        distortion coefficients.
     */
    void undistortKpts();
private: // private data
    std::shared_ptr<cv::Feature2D> mpFeatExtractor; ///< Feature extractor.
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    cv::Mat mDescs; ///< Descriptor of the current frame.
};

} // namespace SLAM_demo

#endif // FRAME_HPP
