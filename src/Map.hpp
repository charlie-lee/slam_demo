/**
 * @file   Map.hpp
 * @brief  Header of map class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#ifndef MAP_HPP
#define MAP_HPP

#include <memory>
#include <set>
#include <vector>

namespace SLAM_demo {

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
    /// Clear the current map.
    void clear();
    /// Remove redundant map points from the map.
    void removeMPts();
private: // private data
    /**
     * @brief Minimum match-to-observation ratio (ratio of the number of times 
     *        being matched to the number of times being observed).
     */
    static const float TH_MIN_RATIO_MATCH_TO_OBS;
    /// Maximum number of frames passed after the map point is seen.
    static const unsigned TH_MAX_NUM_FRMS_LAST_SEEN;
    /// A set of map points.
    std::set<std::shared_ptr<MapPoint>> mspMPts;
private: // private members
    /// Remove a point from the map.
    void removeMPt(std::shared_ptr<MapPoint>& pMPt);
};

} // namespace SLAM_demo

#endif // MAP_HPP
