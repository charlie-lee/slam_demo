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
    void addPt(const std::shared_ptr<MapPoint>& pPt);
    /// Clear the current map.
    void clear();
private: // private data
    /// A set of map points.
    std::set<std::shared_ptr<MapPoint>> mspPts;
private: // private members
    /// Remove a point from the map.
    void removePt(std::shared_ptr<MapPoint>& pPt);
};

} // namespace SLAM_demo

#endif // MAP_HPP
