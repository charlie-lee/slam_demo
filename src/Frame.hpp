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
    /// Relative camera pose \f$T_{cw,k|k-1}\f$ of the current frame.
    CamPose mPose; 
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
    /// Get feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat getFeatDescriptors() const { return mDescs; }
    /// Get frame index of current frame.
    unsigned getFrameIdx() const { return mnIdx; }
    /**
     * @name Coordinate Conversion given Camera Extrinsics and Intrinsics
     * @brief Coordinate conversion among world/cam/image coordinate system.
     * @param[in] Xw \f$3 \times 1\f$ coordinate in world coordinate system.
     * @param[in] Xc \f$3 \times 1\f$ coordinate in camera coordinate system.
     */
    ///@{
    /**
     * @brief World to image coordinate system conversion.
     * @return \f$2 \times 1\f$ image cooordinate w.r.t. the current frame.
     */
    cv::Mat coordWorld2Img(const cv::Mat& Xw) const;
    /**
     * @brief World to camera coordinate system conversion.
     * @return \f$3 \times 1\f$ camera cooordinate w.r.t. the current frame.
     */
    cv::Mat coordWorld2Cam(const cv::Mat& Xw) const;
    /**
     * @brief Camera to image coordinate system conversion.
     * @return \f$2 \times 1\f$ image cooordinate w.r.t. the current frame.
     */
    cv::Mat coordCam2Img(const cv::Mat& Xc) const;
    ///@}
private: // private data
    /// Number of blocks (sub-images) on X direction when extracting features.
    static const int NUM_BLK_X;
    /// Number of blocks (sub-images) on Y direction when extracting features.
    static const int NUM_BLK_Y;
    /// Edge threshold for feature extractor.
    static const int TH_EDGE;
    double mTimestamp; ///< Timestamp info for the current frame.
    unsigned mnIdx; ///< Frame index.
    static unsigned nNextIdx; ///< Frame index for next frame.
    std::shared_ptr<cv::Feature2D> mpFeatExtractor; ///< Feature extractor.
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    /// Feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat mDescs; 
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
