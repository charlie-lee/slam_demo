/**
 * @file   MapPoint.hpp
 * @brief  Header of map point class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#ifndef MAPPOINT_HPP
#define MAPPOINT_HPP

#include <opencv2/core.hpp>

namespace SLAM_demo {

/**
 * @class MapPoint
 * @brief Store related data of each map point in the map of the SLAM system.
 */
class MapPoint {
public: // public members
    /// Default constructor: store empty data.
    MapPoint();
    /**
     * @brief Construct map point with input coordinate and descriptor.
     * @param[in] X      3D world coordinate as a \f$3 \times 1\f$ matrix.
     * @param[in] desc   Descriptor of the map point.
     * @param[in] frmIdx Index of the frame that observed the point.
     */
    MapPoint(const cv::Mat& X, const cv::Mat& desc, int frmIdx);
    /**
     * @name Class Getter/Setters
     * @brief Basic getters/setters for read/update various info on the point.
     */
    ///@{
    /// Get 3D world coordinate.
    cv::Mat getPt() const { return mX; }
    /// Get descriptor of the point.
    cv::Mat getDesc() const { return mDesc; }
    /// Update 3D world coordinate of the point.
    void setPt(const cv::Mat& X) { mX = X.clone(); }
    /// Update descriptor of the point.
    void setDesc(const cv::Mat& desc) { mDesc = desc.clone(); }
    /// Update index of the latest frame that observed the point.
    void setIdxLastObsFrm(int idx) { mnIdxLastObsFrm = idx; }
    ///@}
    /// Update count of being observed by input frames.
    void addCntObs(int n = 1);
    /// Update count of being matched by other 3D points.
    void addCntMatches(int n = 1);
private: // private data
    /// Inhomogeneous 3D world coordinate of the point.
    cv::Mat mX;
    /// Best descriptor of the point.
    cv::Mat mDesc;
    /// Index of the latest frame by which the point is observed.
    int mnIdxLastObsFrm;
    /// Number of times by which the point is observed by input frames.
    int mnCntObs;
    /** 
     * @brief Number of times by which the point is matched with another 
     *        map point after the system is initialized.
     */
    int mnCntMatches;
};

} // namespace SLAM_demo

#endif // MAPPOINT_HPP
