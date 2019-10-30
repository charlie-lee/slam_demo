/**
 * @file   Frame.hpp
 * @brief  Header of Frame class for storing intra- and inter-frame info.
 * @author Charlie Li
 * @date   2019.08.12
 */

#ifndef FRAME_HPP
#define FRAME_HPP

#include "FrameBase.hpp"

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class MapPoint;

/**
 * @class Frame
 * @brief Store intra- and inter-frame info, including keypoint data, 
 *        camera pose, etc.
 */
class Frame : public FrameBase {
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
    /// Get frame index of current frame.
    unsigned index() const { return mnIdx; }
private: // private data
    /// Number of blocks (sub-images) on X direction when extracting features.
    static const int NUM_BLK_X;
    /// Number of blocks (sub-images) on Y direction when extracting features.
    static const int NUM_BLK_Y;
    /// Edge threshold for feature extractor.
    static const int TH_EDGE;
    unsigned mnIdx; ///< Frame index.
    static unsigned nNextIdx; ///< Frame index for next frame.
    std::shared_ptr<cv::Feature2D> mpFeatExtractor; ///< Feature extractor.
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
