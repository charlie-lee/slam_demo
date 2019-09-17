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

#include <opencv2/core.hpp>
#include "Frame.hpp"

namespace SLAM_demo {

MapPoint::MapPoint() : mX3D(cv::Mat()), mDesc(cv::Mat()), mnIdxLastObsFrm(0),
                       mnCntObs(0), mnCntMatches(0), mbOutlier(false) {}

MapPoint::MapPoint(const cv::Mat& X3D, int nIdxFrm) :
    mX3D(X3D.clone()),
    mDesc(cv::Mat()), mnIdxLastObsFrm(nIdxFrm),
    mnCntObs(0), mnCntMatches(0), mbOutlier(false) {}

MapPoint::MapPoint(const cv::Mat& X3D, const cv::Mat& desc, int nIdxFrm) :
    mX3D(X3D.clone()), mDesc(desc.clone()), mnIdxLastObsFrm(nIdxFrm),
    mnCntObs(0), mnCntMatches(0), mbOutlier(false) {}

cv::Mat MapPoint::getDesc(const std::shared_ptr<Frame>& pFrame) const
{
    auto it = mObses.find(pFrame);
    assert(it != mObses.end());
    cv::Mat descs = pFrame->getFeatDescriptors();
    return descs.row(it->second);
}

cv::KeyPoint MapPoint::getKpt(const std::shared_ptr<Frame>& pFrame) const
{
    auto it = mObses.find(pFrame);
    assert(it != mObses.end());
    std::vector<cv::KeyPoint> vKpts = pFrame->getKeyPoints();
    return vKpts[it->second];
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
    mObses.insert(std::make_pair(pFrame, nIdxKpt));
}

void MapPoint::updateDescriptor()
{
    float maxResponse = - std::numeric_limits<float>::max();
    std::shared_ptr<Frame> pFrameBest = nullptr;
    for (auto it = mObses.begin(); it != mObses.end(); ++it) {
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
