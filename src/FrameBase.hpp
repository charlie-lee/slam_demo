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

#include <nanoflann.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <Eigen/Dense>
#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class MapPoint;

/**
 * @class FrameBase
 * @brief Base class for storing data of an input frame.
 */
class FrameBase {
public: // public members
    /// Relative camera pose \f$T_{cw,k|k-1}\f$ of the current frame.
    CamPose mPose; 
public: // public member functions
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
    /// Get keypoint data for current frame given its index.
    cv::KeyPoint keypoint(int nIdx) const { return mvKpts[nIdx]; }
    /// Get feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat descriptors() const { return mDescs; }
    /// Get feature descriptor given its index.
    cv::Mat descriptor(int nIdx) const { return mDescs.row(nIdx); }
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
    /**
     * @brief Get feature indices within a given radius of 2D distance and
     *        a given range of angle.
     * @param[in] xIn        An input point near which the appropriate 
     *                       keypoints are to be found.
     * @param[in] angleIn    The orientation data for the input point.
     * @param[in] radiusDist A radius of distance where the found keypoints 
     *                       should be within.
     * @param[in] angleDiff  Max angle difference to the input keypoint's
     *                       orientation data.
     * @return A vector of indices of valid keypoints.
     */
    std::vector<int> featuresInRange(const cv::Mat& xIn, float angleIn,
                                     float radiusDist, float angleDiff) const;
protected: // protected member functions
    /**
     * @brief Check whether an input angle is within the range of a base angle.
     * @param[in] angleIn   Input angle (degree).
     * @param[in] angleBase Base angle (degree).
     * @param[in] maxDiff   Max angle difference (degree) to the base angle
     * @return True if @p angleIn is within the range of @p angleBase.
     */
    bool isAngleInRange(float angleIn, float angleBase, float maxDiff) const;
protected: // protected members for usage of derived classes
    using nanoflannKDTree =
        nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixX2f,
                                            2, // data dimension
                                            nanoflann::metric_L2_Simple>;
    double mTimestamp; ///< Timestamp info for the current frame.
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    /// Feature descriptors. Row \f$i\f$ for \f$i\f$th descriptor.
    cv::Mat mDescs;
    /// Matched keypoint indices and their corresponding map points.
    std::map<int, std::shared_ptr<MapPoint>> mmpMPts;
    /// Image coordinates (Nx2, Eigen) for keypoints of the current frame.
    std::shared_ptr<Eigen::MatrixX2f> mpx2Ds;
    /// K-D tree of keypoint positions for optimizing keypoint search time.
    std::shared_ptr<nanoflannKDTree> mpKDTree;
};

} // namespace SLAM_demo

#endif // FRAME_BASE_HPP
