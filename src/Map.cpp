/**
 * @file   Map.cpp
 * @brief  Implementation of map class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#include "Map.hpp"

#include <memory>
#include <set>
#include <vector>

#include "MapPoint.hpp"
#include "System.hpp"

namespace SLAM_demo {

using std::vector;
using std::shared_ptr;

const float Map::TH_MIN_RATIO_MATCH_TO_OBS = 0.3f;
const unsigned Map::TH_MAX_NUM_FRMS_LAST_SEEN = 100;

void Map::addMPt(const std::shared_ptr<MapPoint>& pMPt)
{
    mspMPts.insert(pMPt);
}

std::vector<std::shared_ptr<MapPoint>> Map::getAllMPts() const
{
    vector<shared_ptr<MapPoint>> vpMPts;
    vpMPts.reserve(mspMPts.size());
    for (const shared_ptr<MapPoint> pMPt : mspMPts) {
        if (pMPt) {
            vpMPts.push_back(pMPt);
        }
    }
    return vpMPts;
}

void Map::clear()
{
    mspMPts.clear();
}

void Map::removeMPts()
{
    // TODO: add other criterias
    vector<shared_ptr<MapPoint>> vpMPts = getAllMPts();
    for (shared_ptr<MapPoint> pMPt : vpMPts) {
        // target map point may have already removed
        if (!pMPt) {
            continue;
        }
        float m2oRatio = pMPt->getMatch2ObsRatio();
        if (m2oRatio > 0 && m2oRatio < TH_MIN_RATIO_MATCH_TO_OBS) {
            // discard low quality map points that has low
            // match-to-observe ratio            
            removeMPt(pMPt);
        } else if (System::nCurrentFrame >
                   pMPt->getIdxLastObsFrm() + TH_MAX_NUM_FRMS_LAST_SEEN) {
            // the map point should be seen not long before
            removeMPt(pMPt);
        } else if (pMPt->isOutlier()) {
            removeMPt(pMPt);
        }
    }
}

void Map::removeMPt(std::shared_ptr<MapPoint>& pMPt)
{
    mspMPts.erase(pMPt);
}

} // namespace SLAM_demo
