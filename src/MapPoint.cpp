/**
 * @file   MapPoint.cpp
 * @brief  Implementation of map point class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#include "MapPoint.hpp"

#include <opencv2/core.hpp>

namespace SLAM_demo {

MapPoint::MapPoint() : mX3D(cv::Mat()), mDesc(cv::Mat()), mnIdxLastObsFrm(0),
                       mnCntObs(0), mnCntMatches(0) {}

MapPoint::MapPoint(const cv::Mat& X3D, const cv::Mat& desc, int nIdxFrm) :
    mX3D(X3D.clone()), mDesc(desc.clone()), mnIdxLastObsFrm(nIdxFrm),
    mnCntObs(0), mnCntMatches(0) {}

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

} // namespace SLAM_demo
