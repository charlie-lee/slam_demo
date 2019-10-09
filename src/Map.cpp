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

#include "Frame.hpp"
#include "MapPoint.hpp"
#include "System.hpp"

namespace SLAM_demo {

using std::set;
using std::shared_ptr;
using std::vector;

const float Map::TH_MIN_RATIO_OBS_TO_VISIBLE = 0.8f;
const unsigned Map::TH_MAX_NUM_FRMS_LAST_SEEN = 20;

void Map::addMPt(const std::shared_ptr<MapPoint>& pMPt)
{
    mspMPts.insert(pMPt);
}

std::vector<std::shared_ptr<MapPoint>> Map::getAllMPts() const
{
    vector<shared_ptr<MapPoint>> vpMPts;
    vpMPts.reserve(mspMPts.size());
    for (const shared_ptr<MapPoint>& pMPt : mspMPts) {
        if (pMPt) {
            vpMPts.push_back(pMPt);
        }
    }
    return vpMPts;
}

std::vector<std::shared_ptr<Frame>> Map::getAllFrames() const
{
    std::vector<std::shared_ptr<Frame>> vpFrames;
    vpFrames.reserve(mmRFrames.size());
    for (auto cit = mmRFrames.cbegin(); cit != mmRFrames.cend(); ++cit) {
        vpFrames.push_back(cit->first);
    }
    return vpFrames;
}

std::vector<std::shared_ptr<Frame>> Map::getLastNFrames(unsigned nFrames) const
{
    // get all frames if the number of frames is too small or too large
    if (nFrames == 0 || nFrames >= mmRFrames.size()) {
        return getAllFrames();
    }
    vector<shared_ptr<Frame>> vpFrames;
    vpFrames.reserve(nFrames);
    // get all frames in another map and sort them by frame index
    std::map<unsigned, shared_ptr<Frame>> mpFrames;
    for (auto it = mmRFrames.begin(); it != mmRFrames.end(); ++it) {
        mpFrames.insert({it->first->getFrameIdx(), it->first});
    }
    // get last N frames
    for (auto it = mpFrames.rbegin(); it != mpFrames.rend(); ++it) {
        vpFrames.push_back(it->second);
    }
    return vpFrames;
}

void Map::updateFrameData(const std::shared_ptr<Frame>& pFrame, int cnt)
{
    auto it = mmRFrames.find(pFrame);
    if (it == mmRFrames.end()) {
        // insert new frame pointer
        assert(cnt == 1);
        mmRFrames.insert({pFrame, cnt});
    } else {
        assert(it->second > 0);
        it->second += cnt;
        // remove redundant frame with no observed map points
        if (it->second == 0) {
            mspFramesOpt.insert(pFrame);
            mmRFrames.erase(pFrame);
        }
    }
}

void Map::clear()
{
    mspMPts.clear();
    mmRFrames.clear();
}

void Map::removeMPts()
{
    // TODO: add other criterias
    vector<shared_ptr<MapPoint>> vpMPts = getAllMPts();
    for (shared_ptr<MapPoint>& pMPt : vpMPts) {
        // target map point may have already removed
        if (!pMPt) {
            continue;
        }
        float o2vRatio = pMPt->getObs2VisibleRatio();
        if (o2vRatio < TH_MIN_RATIO_OBS_TO_VISIBLE) {
            // discard low quality map points that has low
            // observe-to-visible ratio
            removeMPt(pMPt);
        } else if (System::nCurrentFrame >
                   pMPt->getIdxLastObsFrm() + TH_MAX_NUM_FRMS_LAST_SEEN) {
            removeMPt(pMPt);
        } else if (pMPt->isOutlier()) {
            removeMPt(pMPt);
        }
    }
}

std::set<std::shared_ptr<Frame>> Map::transferFramesOpt()
{
    set<shared_ptr<Frame>> spFramesOpt = mspFramesOpt;
    // remove frame data from the map
    mspFramesOpt.clear();
    // transfer the frame data to System
    return spFramesOpt;
}

void Map::removeMPt(std::shared_ptr<MapPoint>& pMPt)
{
    // update data of related frames
    vector<shared_ptr<Frame>> vpRelatedFrames = pMPt->getRelatedFrames();
    for (const auto& pFrame : vpRelatedFrames) {
        updateFrameData(pFrame, -1);
    }
    mspMPts.erase(pMPt);
}

} // namespace SLAM_demo
