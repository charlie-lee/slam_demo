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
#include <opencv2/flann.hpp>
#include "Config.hpp"
#include "MapPoint.hpp"

namespace SLAM_demo {

using std::shared_ptr;
using std::vector;
using cv::Mat;

FrameBase::FrameBase(double timestamp) :
    mTimestamp(timestamp), mpKDTree(nullptr)
{
}

FrameBase::FrameBase(const FrameBase& rhs) :
    mPose(rhs.mPose),
    mTimestamp(rhs.mTimestamp),
    mvKpts(rhs.mvKpts),
    mDescs(rhs.mDescs.clone()), // clone() necessary??
    mmpMPts(rhs.mmpMPts),
    mpx2Ds(rhs.mpx2Ds),
    mpKDTree(rhs.mpKDTree)
{
}

FrameBase& FrameBase::operator=(const FrameBase& rhs)
{
    mPose = rhs.mPose;
    mTimestamp = rhs.mTimestamp;
    mvKpts = rhs.mvKpts;
    mDescs = rhs.mDescs.clone(); // clone() necessary??
    mmpMPts = rhs.mmpMPts;
    mpx2Ds = rhs.mpx2Ds;
    mpKDTree = rhs.mpKDTree;
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

std::vector<int> FrameBase::featuresInRange(const cv::Mat& xIn,
                                            float angleIn,
                                            float radiusDist,
                                            float angleDiff) const
{
    vector<int> vKptIndices;
    int nKpts = mvKpts.size();
    vKptIndices.reserve(mvKpts.size());
    float x = xIn.at<float>(0);
    float y = xIn.at<float>(1);
    bool bUseKDTree = radiusDist > 0.0f;
    if (bUseKDTree) { // use K-D tree for keypoint searching if available
        float arrxIn[2] = {x, y};
        vector<std::pair<long int, float>> vIdxnDist;
        vIdxnDist.reserve(mvKpts.size());
        nanoflann::SearchParams params;
        params.sorted = false;
        // radius search (squared distance)
        int nFound = mpKDTree->index->radiusSearch(
            arrxIn, radiusDist*radiusDist, vIdxnDist, params);
        for (int idx = 0; idx < nFound; ++idx) {
            int i = vIdxnDist[idx].first; // keypoint index
            // check orientation of target keypoint
            //const auto& kpt = mvKpts[i];
            //if (!isAngleInRange(kpt.angle, angleIn, angleDiff)) {
            //    continue;
            //}
            // add valid keypoints to the result vector
            vKptIndices.push_back(i);
        }
    } else {
        // traverse each keypoint for valid ones
        for (int i = 0; i < nKpts; ++i) {
            const auto& kpt = mvKpts[i];
            // check distance between input and target keypoint
            float distSq = (kpt.pt.x - x) * (kpt.pt.x - x) +
                (kpt.pt.y - y) * (kpt.pt.y - y);
            if (distSq > radiusDist * radiusDist) {
                continue; // ignore keypoint too distant from input keypoint
            }
            // check orientation of target keypoint
            if (!isAngleInRange(kpt.angle, angleIn, angleDiff)) {
                continue;
            }
            // add valid keypoints to the result vector
            vKptIndices.push_back(i);
        }
    }
    return vKptIndices;
}

bool FrameBase::isAngleInRange(float angleIn,
                               float angleBase,
                               float maxDiff) const
{
    bool bInRange = false;
    float angleMin = angleBase - maxDiff;
    float angleMax = angleBase + maxDiff;
    // compute 2 ranges
    float range1Min, range1Max, range2Min, range2Max;
    // case 1: [angleMin, 360] V [0, angleMax]
    if (angleMin < 0.0f || angleMax > 360.0f) {
        if (angleMin < 0.0f) {
            angleMin += 360.0f;
        }
        if (angleMax > 360.0f) {
            angleMax -= 360.0f;
        }
        range1Max = 360.0f;
        range2Min = 0.0f;
    } else { // case 2: [angleMin, angleBase) V [angleBase, angleMax]
        range1Max = range2Min = angleBase;
    }
    range1Min = angleMin;
    range2Max = angleMax;
    // check whether input angle is within one of the 2 ranges
    if ((angleIn >= range1Min && angleIn <= range1Max) ||
        (angleIn >= range2Min && angleIn <= range2Max)) {
        bInRange = true;
    } else {
        bInRange = false;
    }
    return bInRange;
}

} // namespace SLAM_demo
