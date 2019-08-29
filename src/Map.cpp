/**
 * @file   Map.cpp
 * @brief  Implementation of map class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#include "Map.hpp"

#include <memory>
#include <set>

#include "MapPoint.hpp"

namespace SLAM_demo {

void Map::addPt(const std::shared_ptr<MapPoint>& pPt)
{
    mspPts.insert(pPt);
}

void Map::clear()
{
    mspPts.clear();
}

void Map::removePt(std::shared_ptr<MapPoint>& pPt)
{
    mspPts.erase(pPt);
}

} // namespace SLAM_demo
