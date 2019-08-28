/**
 * @file   CamPose.cpp
 * @brief  Implementation of CamPose class for camera pose representation.
 * @author Charlie Li
 * @date   2019.08.23
 */

#include "CamPose.hpp"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp> // cv::cv2eigen()
#include <Eigen/Core>
#include <Eigen/Geometry> // for Matrix3f::eulerAngles()

namespace SLAM_demo {

using std::endl;
using cv::Mat;

CamPose::CamPose() : mTcw(Mat(3, 4, CV_32FC1)), mTwc(Mat(3, 4, CV_32FC1))
{
    Mat I = Mat::eye(3, 3, CV_32FC1);
    Mat zero = Mat::zeros(3, 1, CV_32FC1);
    I.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
    zero.copyTo(mTcw.rowRange(0, 3).col(3));
    setPoseInv();
}

CamPose::CamPose(const cv::Mat& Tcw) :
    mTcw(Tcw.clone()), mTwc(Mat(3, 4, CV_32FC1))
{
    setPoseInv();
}

CamPose::CamPose(const cv::Mat& Rcw, const cv::Mat& tcw) :
    mTcw(Mat(3, 4, CV_32FC1)), mTwc(Mat(3, 4, CV_32FC1))
{
    Rcw.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
    tcw.copyTo(mTcw.rowRange(0, 3).col(3));
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

void CamPose::setPose(const cv::Mat& Rcw, const cv::Mat& tcw)
{
    setRotation(Rcw);
    setTranslation(tcw);
    setPoseInv();
}

Eigen::Matrix<float, 3, 1> CamPose::getREulerAngleEigen() const
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

CamPose& CamPose::operator*(const CamPose& rhs)
{
    Mat RcwL = getRotation();
    Mat tcwL = getTranslation();
    Mat RcwR = rhs.getRotation();
    Mat tcwR = rhs.getTranslation();
    // (4*4 matrix) T_L * T_R = [R_L*R_R, R_L*t_R + t_L; 0^T, 1] 
    setPose(RcwL*RcwR, RcwL*tcwR + tcwL);
    return *this;
}

void CamPose::setRotation(const cv::Mat& Rcw)
{
    Rcw.copyTo(mTcw.rowRange(0, 3).colRange(0, 3));
}

void CamPose::setTranslation(const cv::Mat& tcw)
{
    tcw.copyTo(mTcw.rowRange(0, 3).col(3));
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

Eigen::Matrix<float, 3, 3> CamPose::getRotationEigen() const
{
    Eigen::Matrix<float, 3, 3> R;
    cv::cv2eigen(getRotation(), R);
    return R;
}

std::ostream& operator<<(std::ostream& os, const CamPose& pose)
{
    //os << "Pose Tcw = [Rcw | tcw] = " << endl << pose.getPose() << endl;
    Eigen::Vector3f ea = pose.getREulerAngleEigen();
    os << "Camera origin = " << pose.getCamOrigin().t() << endl;
    os << "Rotation {yaw, pitch, roll} = {"
       << ea(0) << ", " << ea(1) << ", " << ea(2) << "} (deg)" << endl;
    os << "Translation tcw = " << pose.getTranslation().t();
    return os;
}

} // namespace SLAM_demo
