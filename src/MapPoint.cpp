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

namespace SLAM_demo {

using std::shared_ptr;
using std::vector;

MapPoint::MapPoint() : mX3D(cv::Mat()), mDesc(cv::Mat()), mnIdxLastObsFrm(0),
                       mnCntObs(0), mnCntMatches(0), mbOutlier(false) {}

MapPoint::MapPoint(const cv::Mat& X3D, int nIdxFrm) :
    mX3D(X3D.clone()),
    mDesc(cv::Mat()), mnIdxLastObsFrm(nIdxFrm),
    mnCntObs(1), mnCntMatches(1), mbOutlier(false) {}

MapPoint::MapPoint(const cv::Mat& X3D, const cv::Mat& desc, int nIdxFrm) :
    mX3D(X3D.clone()), mDesc(desc.clone()), mnIdxLastObsFrm(nIdxFrm),
    mnCntObs(1), mnCntMatches(1), mbOutlier(false) {}

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

float MapPoint::getMatch2ObsRatio() const
{
    if (mnCntObs == 0) {
        return 0.f;
    } else {
        return static_cast<float>(mnCntMatches) / mnCntObs;
    }
}

void MapPoint::addCntObs(int n)
{
    mnCntObs += n;
}

void MapPoint::addCntMatches(int n)
{
    mnCntMatches += n;
}

void MapPoint::addObservation(const std::shared_ptr<Frame>& pFrame, int nIdxKpt)
{
    mmObses.insert(std::make_pair(pFrame, nIdxKpt));
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

} // namespace SLAM_demo
