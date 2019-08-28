/**
 * @file   MapPoint.cpp
 * @brief  Implementation of map point class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.28
 */

#include "MapPoint.hpp"

#include <opencv2/core.hpp>

namespace SLAM_demo {

MapPoint::MapPoint() : mX(cv::Mat()), mDesc(cv::Mat()), mnIdxLastObsFrm(0),
                       mnCntObs(0), mnCntMatches(0) {}

MapPoint::MapPoint(const cv::Mat& X, const cv::Mat& desc, int frmIdx) :
    mX(X.clone()), mDesc(desc.clone()), mnIdxLastObsFrm(frmIdx),
    mnCntObs(1), mnCntMatches(0) {}

void MapPoint::addCntObs(int n)
{
    mnCntObs += n;
}

void MapPoint::addCntMatches(int n)
{
    mnCntMatches += n;
}

} // namespace SLAM_demo
