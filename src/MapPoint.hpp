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
    MapPoint(const cv::Mat& X3D, int nIdxFrm);
    /** (OBSOLETE)
     * @brief Construct map point with input coordinate and descriptor.
     * @param[in] X3D     3D world coordinate as a \f$3 \times 1\f$ matrix.
     * @param[in] desc    Descriptor of the map point.
     * @param[in] nIdxFrm Index of the frame that observed the point.
     */
    MapPoint(const cv::Mat& X3D, const cv::Mat& desc, int nIdxFrm);
    /// Get 3D world coordinate.
    cv::Mat getX3D() const { return mX3D; }
    /// Get most distinctive descriptor of the point.
    cv::Mat getDesc() const { return mDesc; }
    /// Get descriptor from a frame.
    cv::Mat getDesc(const std::shared_ptr<Frame>& pFrame) const;
    /// Get keypoint data from a frame.
    cv::KeyPoint getKpt(const std::shared_ptr<Frame>& pFrame) const;
    /// Get all related frames that observed it.
    std::vector<std::shared_ptr<Frame>> getRelatedFrames() const;
    /// Get index of the latest frame that observed the point.
    unsigned getIdxLastObsFrm() const { return mnIdxLastObsFrm; }
    /// Get match-to-observation ratio.
    float getMatch2ObsRatio() const;
    /// Update 3D world coordinate of the point.
    void setX3D(const cv::Mat& X3D) { mX3D = X3D.clone(); }
    /// Update descriptor of the point.
    void setDesc(const cv::Mat& desc) { mDesc = desc.clone(); }
    /// Update index of the latest frame that observed the point.
    void setIdxLastObsFrm(int idx) { mnIdxLastObsFrm = idx; }
    /// Update count of being observed by input frames.
    void addCntObs(int n = 1);
    /// Update count of being matched by other points.
    void addCntMatches(int n = 1);
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
    void updateDescriptor();
private: // private data
    /// Inhomogeneous 3D world coordinate of the point.
    cv::Mat mX3D;
    /// Observation data (pointer to frame, index of the observed keypoint).
    std::map<std::shared_ptr<Frame>, int> mmObses;
    /// Best descriptor of the point. (OBSOLETE)
    cv::Mat mDesc;
    /// Index of the latest frame by which the point is observed.
    unsigned mnIdxLastObsFrm;
    /// Number of times by which the point is observed by input frames.
    int mnCntObs;
    /** 
     * @brief Number of times by which the point is matched with another 
     *        point (2D/3D) after the system is initialized.
     */
    int mnCntMatches;
    /// Whether the map point is an outlier.
    bool mbOutlier;
};

} // namespace SLAM_demo

#endif // MAPPOINT_HPP
