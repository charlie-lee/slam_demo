/**
 * @file   Utility.cpp
 * @brief  Implentation of utility class in SLAM system for general functions.
 * @author Charlie Li
 * @date   2019.10.25
 */

#include "Utility.hpp"

#include <memory>
#include <vector>

#include <opencv2/calib3d.hpp> // triangulation, ...
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "CamPose.hpp"
#include "Config.hpp"
#include "FrameBase.hpp"

namespace SLAM_demo {

using cv::Mat;
using std::shared_ptr;
using std::vector;

bool Utility::is2DPtInBorder(const cv::Mat& pt)
{
    bool result =
        (pt.at<float>(0) >= 0 && pt.at<float>(0) < Config::width()) &&
        (pt.at<float>(1) >= 0 && pt.at<float>(1) < Config::height());
    return result;
}

std::vector<cv::Mat> Utility::triangulate3DPts(
    const std::shared_ptr<FrameBase>& pF2,
    const std::shared_ptr<FrameBase>& pF1,
    const std::vector<cv::DMatch>& vMatches21)
{
    if (vMatches21.empty()) {
        return vector<Mat>();
    }
    vector<Mat> vXws;
    vXws.reserve(vMatches21.size());
    // get all 2D-to-2D matches
    const vector<cv::KeyPoint>& vKpts1 = pF1->keypoints();
    const vector<cv::KeyPoint>& vKpts2 = pF2->keypoints();
    int nMatches = vMatches21.size();
    vector<cv::Point2f> vPts1(nMatches);
    vector<cv::Point2f> vPts2(nMatches);
    for (int i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[vMatches21[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[vMatches21[i].queryIdx];
        vPts1[i] = kpt1.pt;
        vPts2[i] = kpt2.pt;
    }
    // get transformation matrix K[R|t] for both views
    Mat Tcw1 = Config::K() * pF1->mPose.getPose();
    Mat Tcw2 = Config::K() * pF2->mPose.getPose();
    // compute triangulated 3D world points (3xN)
    Mat Xw4Ds(4, nMatches, CV_32FC1);
    cv::triangulatePoints(Tcw1, Tcw2, vPts1, vPts2, Xw4Ds);
    Xw4Ds.convertTo(Xw4Ds, CV_32FC1);
    Mat Xws = Xw4Ds.rowRange(0, 3); // 3D points
    for (int i = 0; i < nMatches; ++i) {
        Xws.col(i) /= Xw4Ds.at<float>(3, i);
        const cv::KeyPoint& kpt1 = vKpts1[vMatches21[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[vMatches21[i].queryIdx];
        if (checkTriangulatedPt(Xws.col(i), kpt1, kpt2,
                                pF1->mPose, pF2->mPose,
                                0.9995f)) {
            vXws.push_back(Xws.col(i).clone()); // clone() necessary?
        } else {
            vXws.push_back(Mat()); // empty cv::Mat for bad points
        }
    }
    return vXws;
}

bool Utility::checkTriangulatedPt(const cv::Mat& Xw,
                                  const cv::KeyPoint& kpt1,
                                  const cv::KeyPoint& kpt2,
                                  const CamPose& pose1,
                                  const CamPose& pose2,
                                  float thCosParallax)
{
    // 3D cam coord in view 1
    Mat Rcw1 = pose1.getRotation();
    Mat tcw1 = pose1.getTranslation();
    Mat Xc1 = Rcw1*Xw + tcw1;
    // 3D cam coord in view 2
    Mat Rcw2 = pose2.getRotation();
    Mat tcw2 = pose2.getTranslation();
    Mat Xc2 = Rcw2*Xw + tcw2;
    // condition 1: must be positive depth in both views
    float invDepth1 = 1.f / Xc1.at<float>(2);
    float invDepth2 = 1.f / Xc2.at<float>(2);
    if (invDepth1 <= 0 || invDepth2 <= 0) { 
        return false;
    }
    // condition 2: the parallax of 2 views must not be too small
    float cosParallax = computeCosParallax(Xw, pose1, pose2);
    if (cosParallax > thCosParallax) {
        return false;
    }
    // condition 3: reprojected 2D point needs to be inside image border
    const Mat K = Config::K();
    // projected cam coords in previous and current frames
    Mat Xc1Proj = invDepth1 * K * Xc1; 
    Mat Xc2Proj = invDepth2 * K * Xc2;
    // reprojected 2D image coords in both frames
    Mat x1Reproj = Xc1Proj.rowRange(0, 2);
    Mat x2Reproj = Xc2Proj.rowRange(0, 2);
    if (!is2DPtInBorder(x1Reproj) || !is2DPtInBorder(x2Reproj)) {
        return false;
    }
    // condition 4: reprojection error needs to be lower than a threshold
    //              for both views
    Mat x1 = Mat(kpt1.pt);
    Mat x2 = Mat(kpt2.pt);
    // reprojection error threshold: s^o (d <= s^o)
    float s = Config::scaleFactor();
    int o1 = kpt1.octave;
    int o2 = kpt2.octave;
    float th1Reproj = std::pow(s, o1);
    float th2Reproj = std::pow(s, o2);
    float err1Reproj = cv::norm(x1 - x1Reproj, cv::NORM_L2);
    float err2Reproj = cv::norm(x2 - x2Reproj, cv::NORM_L2);
    if (err1Reproj > th1Reproj || err2Reproj > th2Reproj) {
        return false;
    }
    // condition 5: the keypoint pair should meet epipolar constraint
    Mat F21 = computeFundamental(pose1, pose2);
    Mat F12 = F21.t();
    // mean square symmetric error
    Mat x1h = (cv::Mat_<float>(3, 1) << kpt1.pt.x, kpt1.pt.y, 1.f);
    Mat x2h = (cv::Mat_<float>(3, 1) << kpt2.pt.x, kpt2.pt.y, 1.f);
    float errorF2 = computeReprojErr(F21, F12, x1h, x2h, ReprojErrScheme::F);
    if (errorF2*2.0f > (th1Reproj*th1Reproj + th2Reproj*th2Reproj)) {
        return false;
    }
    return true; // triangulated result is good if all conditions are met    
}

cv::Mat Utility::computeFundamental(const CamPose& CP1,
                                    const CamPose& CP2)
{
    Mat F21;
    // T_{n|m} = T_{n|1} T_{1|m} = T_{n|1} T_{m|1}^{-1}
    CamPose CP1inv = CP1.getCamPoseInv();
    CamPose CP21 = CP2 * CP1inv;
    // F21 = K2^{-T} [t21]_x R21 K1^{-1}
    Mat Kinv = Config::K().inv();
    Mat R21 = CP21.getRotation();
    Mat t21x = CP21.getTranslationSS();
    F21 = Kinv.t() * t21x * R21 * Kinv;
    return F21;
}

float Utility::computeReprojErr(const cv::Mat& T21, const cv::Mat& T12,
                                const cv::Mat& p1, const cv::Mat& p2,
                                ReprojErrScheme eScheme)
{
    // compute reprojection errors for F & H result:
    // error_F = d(x_{2,i}, F_{21} x_{1,i})^2 + d(x_{1,i}, F_{21}^T x_{2,i})^2
    // error_H = d(x_{1,i}, H_{21} x_{1,i})^2 + d(x_{2,i}, H21^{-1} x_{2,i})^2
    float err = 0.0f;
    if (eScheme == ReprojErrScheme::F) {
        // reprojection error for fundamental matrix
        // epipolar line in current frame w.r.t. 2D point in previous frame
        Mat l2 = T21 * p1;
        float al2 = l2.at<float>(0, 0);
        float bl2 = l2.at<float>(1, 0);
        // epipolar line in previous frame w.r.t. 2D point in current frame
        Mat l1 = T12 * p2;
        float al1 = l1.at<float>(0, 0);
        float bl1 = l1.at<float>(1, 0);
        // dist between point (x,y) and line ax+by+c=0: |ax+by+c|/sqrt(a^2+b^2)
        float numerF21 = p2.dot(l2);
        float numerF12 = p1.dot(l1);
        float diffF21Sq = numerF21*numerF21 / (al2*al2 + bl2*bl2);
        float diffF12Sq = numerF12*numerF12 / (al1*al1 + bl1*bl1);
        err = diffF21Sq + diffF12Sq;
    } else if (eScheme == ReprojErrScheme::H) {
        // reprojection error for homography
        Mat p22D = p2.rowRange(0, 2) / p2.at<float>(2);
        Mat p2Reproj = T21 * p1;
        Mat p2Reproj2D = p2Reproj.rowRange(0, 2) / p2Reproj.at<float>(2);
        Mat diffH21 = p22D - p2Reproj2D;
        Mat p12D = p1.rowRange(0, 2) / p1.at<float>(2);
        Mat p1Reproj = T12 * p2;
        Mat p1Reproj2D = p1Reproj.rowRange(0, 2) / p1Reproj.at<float>(2);
        Mat diffH12 = p12D - p1Reproj2D;
        err = diffH21.dot(diffH21) + diffH12.dot(diffH12);
    } else {
        assert(0); // TODO: add other schemes if necessary
    }
    // return mean symmetric reprojection error
    return err / 2.0f;
}

float Utility::computeCosParallax(const cv::Mat& Xw,
                                  const CamPose& pose1, const CamPose& pose2)
{
    // 3D cam coord in view 1
    Mat Rcw1 = pose1.getRotation();
    Mat tcw1 = pose1.getTranslation();
    Mat Xc1 = Rcw1*Xw + tcw1;
    // 3D cam coord in view 2
    Mat Rcw2 = pose2.getRotation();
    Mat tcw2 = pose2.getTranslation();
    Mat Xc2 = Rcw2*Xw + tcw2;
    // camera origins
    Mat O1 = pose1.getCamOrigin(); // camera origin in frame 1
    Mat O2 = pose2.getCamOrigin(); // camera origin in frame 2
    // rays
    Mat Xc1O1 = O1 - Xc1; // vector from Xc1 to O1
    Mat Xc2O2 = O2 - Xc2; // vector from Xc2 to O2
    // norms
    float normXc1O1 = cv::norm(Xc1O1, cv::NORM_L2); // ||Xc1O1||_l2
    float normXc2O2 = cv::norm(Xc2O2, cv::NORM_L2); // ||Xc2O2||_l2
    // parallax computation
    float cosParallax = Xc1O1.dot(Xc2O2) / (normXc1O1 * normXc2O2);
    return cosParallax;
}

} // namespace SLAM_demo
