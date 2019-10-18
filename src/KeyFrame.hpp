/**
 * @file   KeyFrame.hpp
 * @brief  Header of KeyFrame class for storing keyframe info.
 * @author Charlie Li
 * @date   2019.10.18
 */

#ifndef KEYFRAME_HPP
#define KEYFRAME_HPP

#include <memory>
#include <vector>

#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class Frame;
class MapPoint;

/**
 * @class KeyFrame
 * @brief Store keyframe info for local mapper and tracker.
 */
class KeyFrame {
public: // public data
    /// Relative camera pose \f$T_{cw,k|k-1}\f$ of the current frame.
    CamPose mPose; 
public: // public members
    /**
     * @brief Constructor of the KeyFrame class. Construct keyframe from
     *        normal frames.
     * @param[in] pFrame Pointer to the frame.
     */
    KeyFrame(const std::shared_ptr<Frame>& pFrame);
    /// Get keypoint data for current frame.
    std::vector<cv::KeyPoint> keypoints() const { return mvKpts; }
    /// Get feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat descriptors() const { return mDescs; }
    /// Get frame index of current frame.
    unsigned index() const { return mnIdx; }
    /// Get timestamp info of the frame.
    double timestamp() const { return mTimestamp; }
    /// Get all corresponding map points (including unmatched ones).
    std::vector<std::shared_ptr<MapPoint>> MPts() const { return mvpMPts; }
    /// Get number of observed map points
    int getNumMPts() const { return mnMPts; }
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
    /// Bind map point data to keypoint of a specific index.
    void bindMPt(const std::shared_ptr<MapPoint>& pMPt, int idxKpt);
private: // private data
    double mTimestamp; ///< Timestamp info for the current frame.
    unsigned mnIdx; ///< Frame index.
    unsigned mnKFIdx; ///< Keyframe index.
    static unsigned nNextKFIdx; ///< Next keyframe index
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    /// Feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat mDescs;
    /// Matched map points.
    std::vector<std::shared_ptr<MapPoint>> mvpMPts;
    /// Number of observed map points.
    int mnMPts;    
};

} // namespace SLAM_demo

#endif // KEYFRAME_HPP
