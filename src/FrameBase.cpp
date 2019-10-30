/**
 * @file   FrameBase.cpp
 * @brief  Implementation of FrameBase class for basic frame information.
 * @author Charlie Li
 * @date   2019.10.24
 */

#include "FrameBase.hpp"

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "Config.hpp"
#include "MapPoint.hpp"

namespace SLAM_demo {

using std::shared_ptr;
using std::vector;
using cv::Mat;

FrameBase::FrameBase(double timestamp) :
    mTimestamp(timestamp)
{
}

FrameBase::FrameBase(const FrameBase& rhs) :
    mPose(rhs.mPose),
    mTimestamp(rhs.mTimestamp),
    mvKpts(rhs.mvKpts),
    mDescs(rhs.mDescs.clone()), // clone() necessary??
    mmpMPts(rhs.mmpMPts)
{
}

FrameBase& FrameBase::operator=(const FrameBase& rhs)
{
    mPose = rhs.mPose;
    mTimestamp = rhs.mTimestamp;
    mvKpts = rhs.mvKpts;
    mDescs = rhs.mDescs.clone(); // clone() necessary??
    mmpMPts = rhs.mmpMPts;
    return *this;
}

std::map<int, std::shared_ptr<MapPoint>> FrameBase::getMPtsMap() const
{
    return mmpMPts;
}

std::vector<std::shared_ptr<MapPoint>> FrameBase::mappoints() const
{
    vector<shared_ptr<MapPoint>> vpMPts;
    vpMPts.reserve(mmpMPts.size());
    for (const auto& pair : mmpMPts) {
        vpMPts.push_back(pair.second);
    }
    return vpMPts;
}

std::shared_ptr<MapPoint> FrameBase::mappoint(int nIdx) const
{
    auto it = mmpMPts.find(nIdx);
    //assert(it != mmpMPts.end());
    //return it->second;
    if (it == mmpMPts.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

void FrameBase::bindMPt(const std::shared_ptr<MapPoint>& pMPt, int idxKpt)
{
    if (!pMPt && (mmpMPts.find(idxKpt) != mmpMPts.end())) {
        mmpMPts.erase(idxKpt);
    } else {
        mmpMPts[idxKpt] = pMPt;
    }
}

cv::Mat FrameBase::coordWorld2Img(const cv::Mat& Xw) const
{
    Mat Xc = coordWorld2Cam(Xw);
    Mat x = coordCam2Img(Xc);
    return x;
}

cv::Mat FrameBase::coordWorld2Cam(const cv::Mat& Xw) const
{
    Mat Xc(3, 1, CV_32FC1);
    Mat Rcw = mPose.getRotation();
    Mat tcw = mPose.getTranslation();
    Xc = Rcw*Xw + tcw;
    return Xc;
}

cv::Mat FrameBase::coordCam2Img(const cv::Mat& Xc) const
{
    Mat x(2, 1, CV_32FC1);
    Mat K = Config::K();
    float invZc = 1.0f / Xc.at<float>(2);
    Mat x3D = invZc * K * Xc;
    x3D.rowRange(0, 2).copyTo(x);
    return x;
}

} // namespace SLAM_demo
