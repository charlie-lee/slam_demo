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
#include "KeyFrame.hpp"
#include "Map.hpp"
#include "System.hpp"

namespace SLAM_demo {

using std::shared_ptr;
using std::vector;

MapPoint::MapPoint() :
    mpMap(nullptr), mX3D(cv::Mat()),
    mDesc(cv::Mat()), mAngle(-1.0f), mnIdxLastVisibleFrm(0),
    mnCntVisible(0), mnCntTracked(0), mbOutlier(false) {}

MapPoint::MapPoint(const std::shared_ptr<Map>& pMap, const cv::Mat& X3D) :
    mpMap(pMap), mX3D(X3D.clone()),
    mDesc(cv::Mat()), mAngle(-1.0f), mnIdxLastVisibleFrm(0), // set frame index
    mnCntVisible(1), mnCntTracked(1), mbOutlier(false) {}

cv::Mat MapPoint::descriptor(const std::shared_ptr<KeyFrame>& pKF) const
{
    auto it = mmObses.find(pKF);
    assert(it != mmObses.end());
    cv::Mat descs = pKF->descriptors();
    return descs.row(it->second);
}

int MapPoint::keypointIdx(const std::shared_ptr<KeyFrame>& pKF) const
{
    auto it = mmObses.find(pKF);
    assert(it != mmObses.end());
    return it->second;
}

cv::KeyPoint MapPoint::keypoint(const std::shared_ptr<KeyFrame>& pKF) const
{
    vector<cv::KeyPoint> vKpts = pKF->keypoints();
    return vKpts[keypointIdx(pKF)];
}

std::vector<std::shared_ptr<KeyFrame>> MapPoint::getRelatedKFs() const
{
    vector<shared_ptr<KeyFrame>> vpRelatedKFs;
    vpRelatedKFs.reserve(mmObses.size());
    for (auto cit = mmObses.cbegin(); cit != mmObses.cend(); ++cit) {
        vpRelatedKFs.push_back(cit->first);
    }
    return vpRelatedKFs;
}

float MapPoint::getTracked2VisibleRatio() const
{
    if (mnCntVisible == 0) {
        return 0.f;
    } else {
        return static_cast<float>(mnCntTracked) / mnCntVisible;
    }
}


void MapPoint::addCntVisible(int n)
{
    mnCntVisible += n;
}

void MapPoint::addCntTracked(int n)
{
    mnCntTracked += n;
}

void MapPoint::addObservation(const std::shared_ptr<KeyFrame>& pKF, int nIdxKpt)
{
    // add observation count only once for each frame
    if (!isObservedBy(pKF)) {
        mpMap->updateKFData(pKF, 1);
    }
    mmObses.insert(std::make_pair(pKF, nIdxKpt));
}

void MapPoint::removeObservation(const std::shared_ptr<KeyFrame>& pKF)
{
    if (isObservedBy(pKF)) {
        mpMap->updateKFData(pKF, -1);
        mmObses.erase(pKF);
    }
}

void MapPoint::updateDescriptor()
{
    float maxResponse = - std::numeric_limits<float>::max();
    shared_ptr<KeyFrame> pKFBest = nullptr;
    for (auto cit = mmObses.cbegin(); cit != mmObses.cend(); ++cit) {
        // get keypoint & descriptor data
        cv::KeyPoint kpt = this->keypoint(cit->first);
        cv::Mat desc = this->descriptor(cit->first);
        if (kpt.response > maxResponse) {
            maxResponse = kpt.response;
            pKFBest = cit->first;
        }
    }
    // invalid map point without observation data
    if (!pKFBest) {
        return;
    }
    mDesc = descriptor(pKFBest);
    // assign orientation data
    cv::KeyPoint kptBest = keypoint(pKFBest);
    mAngle = kptBest.angle;
}

bool MapPoint::isObservedBy(const std::shared_ptr<KeyFrame>& pKF) const
{
    return (mmObses.find(pKF) != mmObses.end());
}

} // namespace SLAM_demo
