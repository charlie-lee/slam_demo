/**
 * @file   KeyFrame.cpp
 * @brief  Implementation of KeyFrame class for storing keyframe info.
 * @author Charlie Li
 * @date   2019.10.18
 */

#include "KeyFrame.hpp"

#include <algorithm> // std::sort
#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include "Config.hpp"
#include "Frame.hpp"
#include "MapPoint.hpp"

namespace SLAM_demo {

using std::map;
using std::multimap;
using std::shared_ptr;
using std::vector;
using cv::Mat;

unsigned KeyFrame::nNextIdx = 0;
const int KeyFrame::TH_MIN_WEIGHT = 10;

KeyFrame::KeyFrame(const std::shared_ptr<Frame>& pFrame) :
    FrameBase(*pFrame), mnIdx(nNextIdx++), mnIdxFrame(pFrame->index()) {}

void KeyFrame::updateConnections()
{
    // reset std::map data
    mmpKFnWt.clear();
    mvpWeakKFs.clear();
    // temp map for all related data
    map<shared_ptr<KeyFrame>, int> mpKFnWt;
    // traverse each map point (including nullptrs) for KF observations
    for (auto cit = mmpMPts.cbegin(); cit != mmpMPts.cend();) {
        shared_ptr<MapPoint> pMPt = cit->second;
        if (!pMPt) { // remove invalid map point from the map
            cit = mmpMPts.erase(cit);
            continue;
        }
        vector<shared_ptr<KeyFrame>> vpKFs = pMPt->getRelatedKFs();
        for (const auto& pKF : vpKFs) {
            if (pKF->index() == this->index()) { // skip current keyframe
                continue;
            } else {
                mpKFnWt[pKF]++;
            }
        }
        ++cit;
    }
    // filter out those strong links and weak links
    mvpWeakKFs.reserve(mpKFnWt.size());
    for (map<shared_ptr<KeyFrame>, int>::const_iterator cit = mpKFnWt.cbegin();
         cit != mpKFnWt.cend();
         ++cit) {
        if (cit->second >= TH_MIN_WEIGHT) {
            mmpKFnWt[cit->first] = cit->second;
        } else {
            mvpWeakKFs.push_back(cit->first);
        }
    }
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::getConnectedKFs() const
{
    vector<shared_ptr<KeyFrame>> vpKFs;
    vpKFs.reserve(mmpKFnWt.size());
    for (const auto& pair : mmpKFnWt) {
        vpKFs.push_back(pair.first);
    }
    return vpKFs;
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::getWeakKFs() const
{
    return mvpWeakKFs;
}

std::vector<std::shared_ptr<KeyFrame>> KeyFrame::getBestConnectedKFs(int nKFs)
    const
{
    vector<shared_ptr<KeyFrame>> vpKFs;
    // confirm number of returned keyframes
    int nRetKFs;
    int nMaxKFs = mmpKFnWt.size();
    if (nKFs <= 0 || nKFs > nMaxKFs) { // too few / too large an amount
        nRetKFs = nMaxKFs;
    } else {
        nRetKFs = nKFs;
    }
    vpKFs.reserve(nRetKFs);
    // traverse each pair in the std::map and put them in decending order
    // TODO: use multimap!!!
    multimap<int, shared_ptr<KeyFrame>, std::greater<int>> mpWeightedKFs;
    map<shared_ptr<KeyFrame>, int>::const_iterator cit = mmpKFnWt.cbegin();
    for (int i = 0; i < nRetKFs; ++i, ++cit) {
        mpWeightedKFs.insert({cit->second, cit->first});
    }
    // copy the result to std::vector
    for (const auto& pair : mpWeightedKFs) {
        vpKFs.push_back(pair.second);
    }
    return vpKFs;
}
    
} // namespace SLAM_demo
