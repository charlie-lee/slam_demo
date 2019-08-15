/**
 * @file   Tracker.cpp
 * @brief  Implementations of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#include "Tracker.hpp"

#include <memory>
#include <vector>

#include <opencv2/calib3d.hpp> // cv::undistort(), H & F computation, ...
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // cv::imshow()
#include "Config.hpp"
#include "Frame.hpp"

#include <iostream> // temp for debugging

namespace SLAM_demo {

using std::shared_ptr;
using std::make_shared;
using std::vector;
using cv::Mat;

using std::cout;
using std::endl;

Tracker::Tracker(System::Mode eMode) : meMode(eMode), mbFirstFrame(true)
{
    // initialize feature matcher
    mpFeatMatcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);
    // allocate space for vectors
    if (meMode == System::Mode::MONOCULAR) {
        mvImgsPrev.resize(1);
        mvImgsCur.resize(1);
        mvpFramesPrev.resize(1);
        mvpFramesCur.resize(1);
    } else if (meMode == System::Mode::STEREO) {
        mvImgsPrev.resize(2);
        mvImgsCur.resize(2);
        mvpFramesPrev.resize(2);
        mvpFramesCur.resize(2);
    } else if (meMode == System::Mode::RGBD) {
        mvImgsPrev.resize(1);
        mvImgsCur.resize(1);
        mvpFramesPrev.resize(2);
        mvpFramesCur.resize(2);
    }
}

void Tracker::trackImgsMono(const Mat& img, double timestamp)
{
    // temp display
    cout << std::fixed << "[Timestamp " << timestamp << "s]" << endl;
    
    // RGB -> Grayscale
    Mat& imgPrev = mvImgsPrev[0];
    Mat& imgCur = mvImgsCur[0];
    imgCur = rgb2Gray(img);
    // initialize each frame
    shared_ptr<Frame> pFrameCur = make_shared<Frame>(Frame(imgCur, timestamp));
    // feature matching
    if (mbFirstFrame) {
        mvpFramesCur[0] = pFrameCur;
        mbFirstFrame = false;
    } else {
        mvpFramesPrev[0] = mvpFramesCur[0];
        mvpFramesCur[0] = pFrameCur;
        if (initializeMap()) {
            ; // TODO: tracking scheme after map is initialized
        }
    }
    imgPrev = imgCur;
}

Mat Tracker::rgb2Gray(const Mat& img) const
{
    Mat imgGray;
    if (img.channels() == 3) {
        cvtColor(img, imgGray, cv::COLOR_RGB2GRAY);
    } else if (img.channels() == 4) {
        cvtColor(img, imgGray, cv::COLOR_RGBA2GRAY);
    } else {
        imgGray = img.clone();
    }
    return imgGray;
}

bool Tracker::initializeMap()
{
    if (meMode == System::Mode::MONOCULAR) {
        const shared_ptr<Frame>& pFPrev = mvpFramesPrev[0];
        const shared_ptr<Frame>& pFCur = mvpFramesCur[0];
        return initializeMapMono(pFPrev, pFCur);
    }
    return false;
}

bool Tracker::initializeMapMono(const shared_ptr<Frame>& pFPrev,
                                const shared_ptr<Frame>& pFCur)
{
    // match features between current (1) and reference (2) frame
    vector<cv::DMatch> vMatches = matchFeatures2Dto2D(pFPrev, pFCur);
    // temp test on display of feature matching result
    displayFeatMatchResult(vMatches, 0, 0);
    // compute fundamental matrix F (from previous (p) to current (c))
    cv::Mat Fcp = computeFundamental(pFPrev, pFCur, vMatches);
    // compute homography H (from previous (p) to current (c))
    cv::Mat Hcp = computeHomography(pFPrev, pFCur, vMatches);
    //cout << "Fcp = " << endl << Fcp << endl << "Hcp = " << endl << Hcp << endl;
    // reconstruct pose from F or H
    if (reconstructPoseFromFH(pFPrev, pFCur, vMatches, Fcp, Hcp)) {
        ; // TODO: initialize 3D map points by triangulation 2D keypoints
    }
    // temp: always return false as the initalization scheme is incomplete
    return false; 
}

vector<cv::DMatch> Tracker::matchFeatures2Dto2D(
    const shared_ptr<Frame>& pFPrev, const shared_ptr<Frame>& pFCur,
    const float TH_DIST) const
{
    vector<cv::DMatch> vMatches;
    vector<vector<cv::DMatch>> vknnMatches;
    mpFeatMatcher->knnMatch(pFPrev->getFeatDescriptors(), // query
                            pFCur->getFeatDescriptors(), // train
                            vknnMatches, 2); // get 2 best matches
    // find good matches using Lowe's ratio test
    const float TH = TH_DIST;
    const vector<cv::KeyPoint>& vKptsP = pFPrev->getKeyPoints();
    const vector<cv::KeyPoint>& vKptsC = pFCur->getKeyPoints();
    vMatches.reserve(vknnMatches.size());
    for (unsigned i = 0; i < vknnMatches.size(); ++i) {
        if (vknnMatches[i][0].distance < TH * vknnMatches[i][1].distance) {
            // filter out-of-border matches
            const cv::KeyPoint& kptP = vKptsP[vknnMatches[i][0].queryIdx];
            const cv::KeyPoint& kptC = vKptsC[vknnMatches[i][0].trainIdx];
            if (isKptInBorder(kptP) && isKptInBorder(kptC)) {
                vMatches.push_back(vknnMatches[i][0]);
            }
        }
    }
    return vMatches;
}

inline bool Tracker::isKptInBorder(const cv::KeyPoint& kpt) const
{
    bool result =
        (kpt.pt.x >= 0 && kpt.pt.x < Config::width()) &&
        (kpt.pt.y >= 0 && kpt.pt.y < Config::height());
    return result;
}

void Tracker::displayFeatMatchResult(const vector<cv::DMatch>& vMatches,
                                     int viewPrev, int viewCur) const
{
    const cv::Mat& imgPrev = mvImgsPrev[viewPrev];
    const cv::Mat& imgCur = mvImgsCur[viewCur];
    Mat imgPrevUnD, imgCurUnD, imgOut;
    // undistort input images
    cv::undistort(imgPrev, imgPrevUnD, Config::K(), Config::distCoeffs());
    cv::undistort(imgCur, imgCurUnD, Config::K(), Config::distCoeffs());
    // display keypoint matches (undistorted) on undistorted images
    cv::drawMatches(imgPrevUnD, mvpFramesPrev[viewPrev]->getKeyPoints(),
                    imgCurUnD, mvpFramesCur[viewCur]->getKeyPoints(),
                    vMatches, imgOut,
                    cv::Scalar({255, 0, 0}), // color for matching line (BGR)
                    cv::Scalar({0, 255, 0})); // color for keypoint (BGR)
    cv::imshow("cam0: Matches between previous/left (left) and "
               "current/right (right) frame", imgOut);
    cv::waitKey();
}

cv::Mat Tracker::computeFundamental(const shared_ptr<Frame>& pFPrev,
                                    const shared_ptr<Frame>& pFCur,
                                    const vector<cv::DMatch>& vMatches) const
{
    Mat Fcp; // fundamental matrix result (from previous (p) to current (c))
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKptsP = pFPrev->getKeyPoints();
    const vector<cv::KeyPoint>& vKptsC = pFCur->getKeyPoints();
    unsigned nMatches = vMatches.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPtsP, vPtsC; 
    vPtsP.reserve(nMatches);
    vPtsC.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kptP = vKptsP[vMatches[i].queryIdx];
        const cv::KeyPoint& kptC = vKptsC[vMatches[i].trainIdx];
        vPtsP.push_back(kptP.pt);
        vPtsC.push_back(kptC.pt);
    }
    // F computation (using RANSAC)
    Fcp = cv::findFundamentalMat(vPtsP, vPtsC, cv::FM_RANSAC, 1., 0.99,
                                 cv::noArray());
    return Fcp;
}

cv::Mat Tracker::computeHomography(const shared_ptr<Frame>& pFPrev,
                                   const shared_ptr<Frame>& pFCur,
                                   const vector<cv::DMatch>& vMatches) const
{
    Mat Hcp; // fundamental matrix result (from previous (p) to current (c))
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKptsP = pFPrev->getKeyPoints();
    const vector<cv::KeyPoint>& vKptsC = pFCur->getKeyPoints();
    unsigned nMatches = vMatches.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPtsP, vPtsC; 
    vPtsP.reserve(nMatches);
    vPtsC.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kptP = vKptsP[vMatches[i].queryIdx];
        const cv::KeyPoint& kptC = vKptsC[vMatches[i].trainIdx];
        vPtsP.push_back(kptP.pt);
        vPtsC.push_back(kptC.pt);
    }
    // H computation (using RANSAC)
    Hcp = cv::findHomography(vPtsP, vPtsC, cv::RANSAC, 1., cv::noArray(),
                             2000, 0.99);
    return Hcp;
}

bool Tracker::reconstructPoseFromFH(const shared_ptr<Frame>& pFPrev,
                                    const shared_ptr<Frame>& pFCur,
                                    const vector<cv::DMatch>& vMatches,
                                    const cv::Mat& Fcp,
                                    const cv::Mat& Hcp) const
{
    // select the better transformation from either F or H
    FHResult resFH = selectFH(pFPrev, pFCur, vMatches, Fcp, Hcp);
    if (resFH == FHResult::F) {
        return reconstructPoseFromF(pFPrev, pFCur, vMatches, Fcp);
    } else if (resFH == FHResult::H) {
        return reconstructPoseFromH(pFPrev, pFCur, vMatches, Hcp);
    } else if (resFH == FHResult::NONE) {
        return false;
    }
    return false;
}

Tracker::FHResult Tracker::selectFH(const shared_ptr<Frame>& pFPrev,
                                    const shared_ptr<Frame>& pFCur,
                                    const vector<cv::DMatch>& vMatches,
                                    const cv::Mat& Fcp,
                                    const cv::Mat& Hcp) const
{
    // compute reprojection errors for F & H result:
    // error_F = sigma_i(d(x_{2,i)^T F_{21} x_{1,i})^2 +
    //                   d(x_{1,i}^T F_{21}^{-1} x_{2,i})^2
    // error_H = sigma_i(d(x_{1,i}, H_{21} x_{1,i})^2 +
    //                   d(x_{2,i}, H21^{-1} x_{2,i})^2)
    return FHResult::NONE;
}

bool Tracker::reconstructPoseFromF(const shared_ptr<Frame>& pFPrev,
                                   const shared_ptr<Frame>& pFCur,
                                   const vector<cv::DMatch>& vMatches,
                                   const cv::Mat& Fcp) const
{
    return false;
}

bool Tracker::reconstructPoseFromH(const shared_ptr<Frame>& pFPrev,
                                   const shared_ptr<Frame>& pFCur,
                                   const vector<cv::DMatch>& vMatches,
                                   const cv::Mat& Hcp) const
{
    return false;
}

} // Namespace SLAM_demo
