/**
 * @file   MapPoint.cpp
 * @brief  Implementation of map point class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#include "MapPoint.hpp"

#include <limits> // std::numeric_limits
#include <map>
#include <memory>
#include <utility> // std::make_pair()
#include <vector>

#include <opencv2/core.hpp>
#include "Frame.hpp"
#include "Map.hpp"
#include "System.hpp"

namespace SLAM_demo {

using std::shared_ptr;
using std::vector;

MapPoint::MapPoint() :
    mpMap(nullptr), mX3D(cv::Mat()),
    mDesc(cv::Mat()), mnIdxLastObsFrm(0),
    mnCntVisible(0), mnCntObs(0), mbOutlier(false) {}

MapPoint::MapPoint(const std::shared_ptr<Map>& pMap, const cv::Mat& X3D) :
    mpMap(pMap), mX3D(X3D.clone()),
    mDesc(cv::Mat()), mnIdxLastObsFrm(System::nCurrentFrame),
    mnCntVisible(2), mnCntObs(0), mbOutlier(false) {}

cv::Mat MapPoint::getDesc(const std::shared_ptr<Frame>& pFrame) const
{
    auto it = mmObses.find(pFrame);
    assert(it != mmObses.end());
    cv::Mat descs = pFrame->getFeatDescriptors();
    return descs.row(it->second);
}

cv::KeyPoint MapPoint::getKpt(const std::shared_ptr<Frame>& pFrame) const
{
    auto it = mmObses.find(pFrame);
    assert(it != mmObses.end());
    vector<cv::KeyPoint> vKpts = pFrame->getKeyPoints();
    return vKpts[it->second];
}

std::vector<std::shared_ptr<Frame>> MapPoint::getRelatedFrames() const
{
    vector<shared_ptr<Frame>> vpRelatedFrames;
    vpRelatedFrames.reserve(mmObses.size());
    for (auto cit = mmObses.cbegin(); cit != mmObses.cend(); ++cit) {
        vpRelatedFrames.push_back(cit->first);
    }
    return vpRelatedFrames;
}

float MapPoint::getObs2VisibleRatio() const
{
    if (mnCntVisible == 0) {
        return 0.f;
    } else {
        return static_cast<float>(mnCntObs) / mnCntVisible;
    }
}

void MapPoint::addCntVisible(int n)
{
    mnCntVisible += n;
}

void MapPoint::addObservation(const std::shared_ptr<Frame>& pFrame, int nIdxKpt)
{
    // add observation count only once for each frame
    if (!isObservedBy(pFrame)) {
        addCntObs(1);
        mpMap->updateFrameData(pFrame, 1);
    }
    mmObses.insert(std::make_pair(pFrame, nIdxKpt));
}

void MapPoint::removeObservation(const std::shared_ptr<Frame>& pFrame)
{
    if (isObservedBy(pFrame)) {
        addCntObs(-1);
        mpMap->updateFrameData(pFrame, -1);
        mmObses.erase(pFrame);
    }
}

void MapPoint::updateDescriptor()
{
    float maxResponse = - std::numeric_limits<float>::max();
    shared_ptr<Frame> pFrameBest = nullptr;
    for (auto it = mmObses.begin(); it != mmObses.end(); ++it) {
        // get keypoint & descriptor data
        cv::KeyPoint kpt = getKpt(it->first);
        cv::Mat desc = getDesc(it->first);
        if (kpt.response > maxResponse) {
            maxResponse = kpt.response;
            pFrameBest = it->first;
        }
    }
    mDesc = getDesc(pFrameBest);
}

bool MapPoint::isObservedBy(const std::shared_ptr<Frame>& pFrame) const
{
    return (mmObses.find(pFrame) != mmObses.end());
}

void MapPoint::addCntObs(int n)
{
    mnCntObs += n;
}

} // namespace SLAM_demo
