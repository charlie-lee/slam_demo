/**
 * @file   MapPoint.hpp
 * @brief  Header of map point class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#ifndef MAPPOINT_HPP
#define MAPPOINT_HPP

#include <map>

#include <opencv2/core.hpp>

namespace SLAM_demo {

// forward declarations
class Frame;
class Map;

/**
 * @class MapPoint
 * @brief Store related data of each map point in the map of the SLAM system.
 */
class MapPoint {
public: // public members
    /// Default constructor: store empty data.
    MapPoint();
    /**
     * @brief Construct map point with input world coordinate.
     * @param[in] X3D 3D world coordinate as a \f$3 \times 1\f$ matrix.
     * @param[in] nIdxFrm Index of the frame that observed the point.
     */
    MapPoint(const std::shared_ptr<Map>& pMap, const cv::Mat& X3D);
    /// Get 3D world coordinate.
    cv::Mat X3D() const { return mX3D; }
    /// Get most distinctive descriptor of the point.
    cv::Mat descriptor() const { return mDesc; }
    /// Get descriptor from a frame.
    cv::Mat descriptor(const std::shared_ptr<Frame>& pFrame) const;
    /// Get keypoint data from a frame.
    cv::KeyPoint keypoint(const std::shared_ptr<Frame>& pFrame) const;
    /// Get all related frames that observed it.
    std::vector<std::shared_ptr<Frame>> getRelatedFrames() const;
    /// Get index of the latest frame that observed the point.
    unsigned getIdxLastObsFrm() const { return mnIdxLastObsFrm; }
    /// Get observation-to-visible ratio.
    float getObs2VisibleRatio() const;
    /// Update 3D world coordinate of the point.
    void setX3D(const cv::Mat& X3D) { mX3D = X3D.clone(); }
    /// Update descriptor of the point.
    //void setDescriptor(const cv::Mat& desc) { mDesc = desc.clone(); }
    /// Update index of the latest frame that observed the point.
    void setIdxLastObsFrm(int idx) { mnIdxLastObsFrm = idx; }
    /// Update count of being observed by input frames.
    void addCntVisible(int n = 1);
    /// Check whether a map point is an outlier.
    bool isOutlier() const { return mbOutlier; }
    /// Set outlier flag.
    void setOutlier(bool bOutlier) { mbOutlier = bOutlier; }
    /**
     * @brief Add new observation data for the map point.
     * @param[in] pFrame  Pointer to the frame that observes the map point.
     * @param[in] nIdxKpt Index of the keypoint extracted from the frame.
     */
    void addObservation(const std::shared_ptr<Frame>& pFrame, int nIdxKpt);
    void removeObservation(const std::shared_ptr<Frame>& pFrame);
    /// Update the best descriptor for the map point.
    void updateDescriptor();
    /// Whether the map point is observed by a frame or not.
    bool isObservedBy(const std::shared_ptr<Frame>& pFrame) const;
private: // private data
    // Pointer to the map.
    std::shared_ptr<Map> mpMap;
    /// Inhomogeneous 3D world coordinate of the point.
    cv::Mat mX3D;
    /// Observation data (pointer to frame, index of the observed keypoint).
    std::map<std::shared_ptr<Frame>, int> mmObses;
    /// Best descriptor of the point. (OBSOLETE)
    cv::Mat mDesc;
    /// Index of the latest frame by which the point is observed.
    unsigned mnIdxLastObsFrm;
    /// Number of times by which the point is observed by input frames.
    int mnCntVisible;
    /** 
     * @brief Number of times by which the point is matched with another 
     *        point (2D/3D) after the system is initialized.
     */
    int mnCntObs;
    /// Whether the map point is an outlier.
    bool mbOutlier;
private: // private members
    /// Update count of being matched by other points.
    void addCntObs(int n = 1);
};

} // namespace SLAM_demo

#endif // MAPPOINT_HPP
