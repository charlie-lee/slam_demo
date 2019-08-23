/**
 * @file   CamPose.cpp
 * @brief  Implementation of CanPose class for camera pose representation.
 * @author Charlie Li
 * @date   2019.08.23
 */

#include "CamPose.hpp"

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry> // for Matrix3f::eulerAngles()

namespace SLAM_demo {

using cv::Mat;

CamPose::CamPose() : mTcw(Mat(3, 4, CV_32FC1)), mTwc(Mat(3, 4, CV_32FC1))
{
    Mat I = Mat::eye(3, 3, CV_32FC1);
    Mat zero = Mat::zeros(3, 1, CV_32FC1);
    I.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
    zero.copyTo(mTcw.rowRange(0, 3).col(3));
    setPoseInv();
}

CamPose::CamPose(const Mat& Tcw): mTcw(Tcw.clone()), mTwc(Mat(3, 4, CV_32FC1))
{
    setPoseInv();
}

CamPose::CamPose(const CamPose& pose)
{
    mTcw = pose.mTcw;
    mTwc = pose.mTwc;
}

CamPose& CamPose::operator=(const CamPose& pose)
{
    mTcw = pose.mTcw;
    mTwc = pose.mTwc;
    return *this;
}

void CamPose::setPose(const cv::Mat& Tcw)
{
    mTcw = Tcw.clone();
    setPoseInv();
}

void CamPose::setRotation(const Mat& Rcw)
{
    Rcw.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
    setPoseInv();
}

void CamPose::setTranslation(const Mat& tcw)
{
    tcw.copyTo(mTcw.rowRange(0, 3).col(3));
    setPoseInv();
}

Eigen::Matrix<float, 3, 1> CamPose::getREulerAngleEigen()
{
    Eigen::Matrix<float, 3, 1> ea;
    Eigen::Matrix<float, 3, 3> R = getRotationEigen();
    // construct corresponding Euler angles from rotation matrix representation
    // (yaw, pitch, roll)^T
    ea = R.eulerAngles(2, 1, 0);
    // convert unit from radian to degree
    ea *= 180.f / M_PI;
    return ea;
}

void CamPose::setPoseInv()
{
    Mat Rcw = getRotation();
    Mat tcw = getTranslation();
    Mat Rwc = Rcw.t();
    Mat twc = -Rwc*tcw;
    Rwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
    twc.copyTo(mTwc.rowRange(0, 3).col(3));    
}

Eigen::Matrix<float, 3, 3> CamPose::getRotationEigen()
{
    Eigen::Matrix<float, 3, 3> R;
    R << mTcw.at<float>(0,0), mTcw.at<float>(0,1), mTcw.at<float>(0,2),
         mTcw.at<float>(1,0), mTcw.at<float>(1,1), mTcw.at<float>(1,2),
         mTcw.at<float>(2,0), mTcw.at<float>(2,1), mTcw.at<float>(2,2);
    return R;
}

} // namespace SLAM_demo
