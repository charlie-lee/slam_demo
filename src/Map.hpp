/**
 * @file   Map.hpp
 * @brief  Header of map class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#ifndef MAP_HPP
#define MAP_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace SLAM_demo {

class KeyFrame;
class MapPoint;

/**
 * @class Map
 * @brief Store the observed map of the SLAM system.
 */
class Map {
public: // public members
    /// Default constructor.
    Map() = default;
    /// Add a point into the map.
    void addMPt(const std::shared_ptr<MapPoint>& pMPt);
    /// Get all map points from the map.
    std::vector<std::shared_ptr<MapPoint>> getAllMPts() const;
    /// Get all related keyframes from the map.
    std::vector<std::shared_ptr<KeyFrame>> getAllKFs() const;
    /// Get last N keyframes. If N <= 0, return all related keyframes.
    std::vector<std::shared_ptr<KeyFrame>> getLastNKFs(unsigned nKFs) const;
    /** 
     * @brief Update map point count data of a related keyframe, and remove 
     *        redundant keyframe with no observed map point.
     * @param[in] pKF Pointer to a keyframe.
     * @param[in] cnt The map point count to be added to the related keyframe.
     * @note @p pKF will be removed from the map if its count becomes 0.
     */
    void updateKFData(const std::shared_ptr<KeyFrame>& pKF, int cnt);
    /// Clear the current map.
    void clear();
    /// Remove redundant map points from the map.
    void removeMPts();
    /// Compute scene median depth of the map.
    float computeMedianDepth() const;
    /// Scale all map points to a target depth.
    void scaleToDepth(float depth) const;
    /** 
     * @brief Transfer keyframes whose poses are optimized and will not receive 
     *        further optimization to the System.
     */
    std::set<std::shared_ptr<KeyFrame>> transferKFsOpt();
private: // private data
    /**
     * @brief Minimum tracked-to-visible ratio (ratio of the number of times
     *        being matched to the number of times being seen).
     */
    static const float TH_MIN_RATIO_TRACKED_TO_VISIBLE;
    /// Maximum number of frames passed after the map point is seen.
    static const unsigned TH_MAX_NUM_FRMS_LAST_SEEN;
    /// A set of map points.
    std::set<std::shared_ptr<MapPoint>> mspMPts;
    /// A map of (related frame, number of observed map points) pairs.
    std::map<std::shared_ptr<KeyFrame>, int> mmpKFs;
    /**
     * @brief A set of frames whose poses are optimized.
     * @note The frames are removed from the map, so no further optimization
     *       is available. After the System fetches these frames, they will
     *       be removed from the Map object.
     */
    std::set<std::shared_ptr<KeyFrame>> mspKFsOpt;
private: // private members
    /// Remove a point from the map.
    void removeMPt(std::shared_ptr<MapPoint>& pMPt);
};

} // namespace SLAM_demo

#endif // MAP_HPP
