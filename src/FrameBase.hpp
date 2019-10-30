/**
 * @file   FrameBase.hpp
 * @brief  Header of FrameNase class for basic frame information.
 * @author Charlie Li
 * @date   2019.10.24
 */

#ifndef FRAME_BASE_HPP
#define FRAME_BASE_HPP

#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class MapPoint;

/**
 * @class FrameBase
 * @brief Base class for storing data of an input frame.
 */
class FrameBase {
public: // public data
    /// Relative camera pose \f$T_{cw,k|k-1}\f$ of the current frame.
    CamPose mPose; 
public: // public members
    /**
     * @brief Constructor of the FrameBase class.
     * @param[in] timestamp Timestamp info of current frame.
     */
    FrameBase(double timestamp);
    /// Copy constructor,
    FrameBase(const FrameBase& rhs);
    /// Copy assignment operator.
    FrameBase& operator=(const FrameBase& rhs);
    /// Base destructor.
    virtual ~FrameBase() = default;
    /// Get keypoint data for current frame.
    std::vector<cv::KeyPoint> keypoints() const { return mvKpts; }
    /// Get keypoint data for current frame.
    cv::KeyPoint keypoint(int nIdx) const { return mvKpts[nIdx]; }
    /// Get feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat descriptors() const { return mDescs; }
    /// Get timestamp info of the frame.
    double timestamp() const { return mTimestamp; }
    /// Get all corresponding matched keypoint indices and map points.
    std::map<int, std::shared_ptr<MapPoint>> getMPtsMap() const;
    /// Get all matched map points.
    std::vector<std::shared_ptr<MapPoint>> mappoints() const;
    /// Get all corresponding map points (including unmatched ones).
    std::shared_ptr<MapPoint> mappoint(int nIdx) const;
    /// Get number of observed map points
    int getNumMPts() const { return mmpMPts.size(); }
    /// Bind map point data to keypoint of a specific index.
    void bindMPt(const std::shared_ptr<MapPoint>& pMPt, int idxKpt);
    /// Reset map point observations.
    void resetMPtObservations() { mmpMPts.clear(); }
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
protected: // protected data for usage of derived classes
    double mTimestamp; ///< Timestamp info for the current frame.
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    /// Feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat mDescs;
    /// Matched keypoint indices and their corresponding map points.
    std::map<int, std::shared_ptr<MapPoint>> mmpMPts;
};

} // namespace SLAM_demo

#endif // FRAME_BASE_HPP
