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

class Frame;
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
    /// Get all related frames from the map.
    std::vector<std::shared_ptr<Frame>> getAllFrames() const;
    /// Get last N frames. If N <= 0, return all related frames.
    std::vector<std::shared_ptr<Frame>> getLastNFrames(unsigned nFrames) const;
    /** 
     * @brief Update map point count data of a related frame, and remove 
     *        redundant frame with no observed map point.
     * @param[in] pFrame Pointer to a frame.
     * @param[in] cnt    The map point count to be added to the related frame.
     * @note @p pFrame will be removed from the map if its count becomes 0.
     */
    void updateFrameData(const std::shared_ptr<Frame>& pFrame, int cnt);
    /// Clear the current map.
    void clear();
    /// Remove redundant map points from the map.
    void removeMPts();
    /** 
     * @brief Transfer frames whose poses are optimized and will not receive 
     *        further optimization to the System.
     */
    std::set<std::shared_ptr<Frame>> transferFramesOpt();
private: // private data
    /**
     * @brief Minimum observe-to-visible ratio (ratio of the number of times
     *        being matched to the number of times being observed).
     */
    static const float TH_MIN_RATIO_OBS_TO_VISIBLE;
    /// Maximum number of frames passed after the map point is seen.
    static const unsigned TH_MAX_NUM_FRMS_LAST_SEEN;
    /// A set of map points.
    std::set<std::shared_ptr<MapPoint>> mspMPts;
    /// A map of (related frame, number of observed map points) pairs.
    std::map<std::shared_ptr<Frame>, int> mmRFrames;
    /**
     * @brief A set of frames whose poses are optimized.
     * @note The frames are removed from the map, so no further optimization
     *       is available. After the System fetches these frames, they will
     *       be removed from the Map object.
     */
    std::set<std::shared_ptr<Frame>> mspFramesOpt;
private: // private members
    /// Remove a point from the map.
    void removeMPt(std::shared_ptr<MapPoint>& pMPt);
};

} // namespace SLAM_demo

#endif // MAP_HPP
