/**
 * @file   MapPoint.hpp
 * @brief  Header of map point class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#ifndef MAPPOINT_HPP
#define MAPPOINT_HPP

#include <map>
#include <memory>

#include <opencv2/core.hpp>

namespace SLAM_demo {

// forward declarations
class KeyFrame;
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
     * @param[in] pMap Pointer to the map.
     * @param[in] X3D  3D world coordinate as a \f$3 \times 1\f$ matrix.
     */
    MapPoint(const std::shared_ptr<Map>& pMap, const cv::Mat& X3D);
    /// Get 3D world coordinate.
    cv::Mat X3D() const { return mX3D; }
    /// Get most distinctive descriptor of the point.
    cv::Mat descriptor() const { return mDesc; }
    /// Get descriptor from a keyframe.
    cv::Mat descriptor(const std::shared_ptr<KeyFrame>& pKF) const;
    /// Get the index of the observed keypoint in a keyframe.
    int keypointIdx(const std::shared_ptr<KeyFrame>& pKF) const;
    /// Get keypoint data from a keyframe.
    cv::KeyPoint keypoint(const std::shared_ptr<KeyFrame>& pKF) const;
    /// Get all related keyframes that observed it.
    std::vector<std::shared_ptr<KeyFrame>> getRelatedKFs() const;
    /// Get index of the latest frame that observed the point.
    unsigned getIdxLastVisibleFrm() const { return mnIdxLastVisibleFrm; }
    /// Get number of observations.
    int getNumObservations() const { return mnCntObs; }
    /// Get observation-to-visible ratio.
    float getTracked2VisibleRatio() const;
    /// Update 3D world coordinate of the point.
    void setX3D(const cv::Mat& X3D) { mX3D = X3D.clone(); }
    /// Update index of the latest frame that observed the point.
    void setIdxLastVisibleFrm(int idx) { mnIdxLastVisibleFrm = idx; }
    /// Update count of being visible in input frames.
    void addCntVisible(int n = 1);
    /// Update count of being tracked by input frames.
    void addCntTracked(int n = 1);
    /// Check whether a map point is an outlier.
    bool isOutlier() const { return mbOutlier; }
    /// Set outlier flag.
    void setOutlier(bool bOutlier) { mbOutlier = bOutlier; }
    /**
     * @brief Add new observation data for the map point.
     * @param[in] pKF     Pointer to the keyframe that observes the map point.
     * @param[in] nIdxKpt Index of the keypoint extracted from the frame.
     */
    void addObservation(const std::shared_ptr<KeyFrame>& pKF, int nIdxKpt);
    /// Remove observation data of the target keyframe.
    void removeObservation(const std::shared_ptr<KeyFrame>& pKF);
    /// Update the best descriptor for the map point.
    void updateDescriptor();
    /// Whether the map point is observed by a frame or not.
    bool isObservedBy(const std::shared_ptr<KeyFrame>& pKF) const;
private: // private data
    // Pointer to the map.
    std::shared_ptr<Map> mpMap;
    /// Inhomogeneous 3D world coordinate of the point.
    cv::Mat mX3D;
    /// Observation data (pointer to frame, index of the observed keypoint).
    std::map<std::shared_ptr<KeyFrame>, int> mmObses;
    /// Best descriptor of the point.
    cv::Mat mDesc;
    /// Index of the latest frame by which the point is tracked.
    unsigned mnIdxLastVisibleFrm;
    /// Number of times by which the point is visible in input frames.
    int mnCntVisible;
    /// Number of times by which the point is tracked by input frames,
    int mnCntTracked;
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
