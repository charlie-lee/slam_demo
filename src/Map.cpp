/**
 * @file   Map.cpp
 * @brief  Implementation of map class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#include "Map.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include "KeyFrame.hpp"
#include "MapPoint.hpp"
#include "System.hpp"

namespace SLAM_demo {

using cv::Mat;
using std::set;
using std::shared_ptr;
using std::vector;

const float Map::TH_MIN_RATIO_TRACKED_TO_VISIBLE = 0.25f;
const unsigned Map::TH_MAX_NUM_FRMS_LAST_SEEN = 10000;

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

std::vector<std::shared_ptr<KeyFrame>> Map::getAllKFs() const
{
    std::vector<std::shared_ptr<KeyFrame>> vpKFs;
    vpKFs.reserve(mmpKFs.size());
    for (auto cit = mmpKFs.cbegin(); cit != mmpKFs.cend(); ++cit) {
        vpKFs.push_back(cit->first);
    }
    return vpKFs;
}

std::vector<std::shared_ptr<KeyFrame>> Map::getLastNKFs(unsigned nKFs) const
{
    // get all keyframes if the number of keyframes is too small or too large
    if (nKFs == 0 || nKFs >= mmpKFs.size()) {
        return getAllKFs();
    }
    vector<shared_ptr<KeyFrame>> vpKFs;
    vpKFs.reserve(nKFs);
    // get all frames in another map and sort them by frame index
    std::map<unsigned, shared_ptr<KeyFrame>> mpKFs;
    for (auto cit = mmpKFs.cbegin(); cit != mmpKFs.cend(); ++cit) {
        mpKFs.insert({cit->first->index(), cit->first});
    }
    // get last N frames
    for (auto it = mpKFs.rbegin(); it != mpKFs.rend(); ++it) {
        vpKFs.push_back(it->second);
    }
    return vpKFs;
}

void Map::updateKFData(const std::shared_ptr<KeyFrame>& pKF, int cnt)
{
    auto it = mmpKFs.find(pKF);
    if (it == mmpKFs.end()) {
        // insert new frame pointer
        assert(cnt == 1);
        mmpKFs.insert({pKF, cnt});
    } else {
        assert(it->second > 0);
        it->second += cnt;
        // remove redundant frame with no observed map points
        if (it->second == 0) {
            mspKFsOpt.insert(pKF);
            mmpKFs.erase(pKF);
        }
    }
}

void Map::clear()
{
    mspMPts.clear();
    mmpKFs.clear();
    mspKFsOpt.clear();
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
        float t2vRatio = pMPt->getTracked2VisibleRatio();
        // target map point has no observations
        if (pMPt->getNumObservations() == 0) {
            removeMPt(pMPt);
        } else if (t2vRatio < TH_MIN_RATIO_TRACKED_TO_VISIBLE) {
            // discard low quality map points that has low
            // tracked-to-visible ratio
            removeMPt(pMPt);
        } else if (System::nCurrentFrame >
                   pMPt->getIdxLastVisibleFrm() + TH_MAX_NUM_FRMS_LAST_SEEN) {
            removeMPt(pMPt);
        } else if (pMPt->isOutlier()) {
            removeMPt(pMPt);
        }
    }
}

float Map::computeMedianDepth() const
{
    vector<float> vDepths;
    vDepths.reserve(mspMPts.size());
    for (const auto& pMPt : mspMPts) {
        Mat Xw = pMPt->X3D();
        float depth = Xw.at<float>(2);
        vDepths.push_back(depth);
    }
    std::sort(vDepths.begin(), vDepths.end());
    return vDepths[(vDepths.size() - 1) / 2];
}

void Map::scaleToDepth(float depth) const
{
    for (const auto& pMPt : mspMPts) {
        Mat Xw = pMPt->X3D();
        Xw /= depth;
        pMPt->setX3D(Xw);
    }    
}

std::set<std::shared_ptr<KeyFrame>> Map::transferKFsOpt()
{
    set<shared_ptr<KeyFrame>> spKFsOpt = mspKFsOpt;
    // remove frame data from the map
    mspKFsOpt.clear();
    // transfer the frame data to System
    return spKFsOpt;
}

void Map::removeMPt(std::shared_ptr<MapPoint>& pMPt)
{
    // update data of related keyframes
    vector<shared_ptr<KeyFrame>> vpRelatedKFs = pMPt->getRelatedKFs();
    for (const auto& pKF : vpRelatedKFs) {
        // unbind map point and the corresponding observed keypoint or each KF
        pKF->bindMPt(nullptr, pMPt->keypointIdx(pKF));
        // update number of observed map point for each keyframe
        updateKFData(pKF, -1);
    }
    mspMPts.erase(pMPt);
}

} // namespace SLAM_demo
