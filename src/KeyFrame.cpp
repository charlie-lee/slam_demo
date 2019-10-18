/**
 * @file   KeyFrame.cpp
 * @brief  Implementation of KeyFrame class for storing keyframe info.
 * @author Charlie Li
 * @date   2019.10.18
 */

#include "KeyFrame.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "Config.hpp"
#include "Frame.hpp"

namespace SLAM_demo {

using std::vector;
using std::shared_ptr;
using cv::Mat;

unsigned KeyFrame::nNextKFIdx = 0;

KeyFrame::KeyFrame(const std::shared_ptr<Frame>& pFrame) :
    mPose(pFrame->mPose),
    mTimestamp(pFrame->timestamp()),
    mnIdx(pFrame->index()),
    mnKFIdx(nNextKFIdx++),
    mvKpts(pFrame->keypoints()),
    mDescs(pFrame->descriptors()),
    mvpMPts(pFrame->MPts()),
    mnMPts(pFrame->getNumMPts())
{
}

cv::Mat KeyFrame::coordWorld2Img(const cv::Mat& Xw) const
{
    Mat Xc = coordWorld2Cam(Xw);
    Mat x = coordCam2Img(Xc);
    return x;
}

cv::Mat KeyFrame::coordWorld2Cam(const cv::Mat& Xw) const
{
    Mat Xc(3, 1, CV_32FC1);
    Mat Rcw = mPose.getRotation();
    Mat tcw = mPose.getTranslation();
    Xc = Rcw*Xw + tcw;
    return Xc;
}

cv::Mat KeyFrame::coordCam2Img(const cv::Mat& Xc) const
{
    Mat x(2, 1, CV_32FC1);
    Mat K = Config::K();
    float invZc = 1.0f / Xc.at<float>(2);
    Mat x3D = invZc * K * Xc;
    x3D.rowRange(0, 2).copyTo(x);
    return x;
}
    
} // namespace SLAM_demo
