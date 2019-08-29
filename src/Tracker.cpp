/**
 * @file   Tracker.cpp
 * @brief  Implementations of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#include "Tracker.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <opencv2/calib3d.hpp> // cv::undistort(), H & F computation, ...
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // cv::imshow()
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include "CamPose.hpp"
#include "Config.hpp"
#include "Frame.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"

#include <iostream> // temp for debugging

namespace SLAM_demo {

using std::shared_ptr;
using std::make_shared;
using std::vector;
using cv::Mat;

using std::cout;
using std::endl;

// constants
const float Tracker::TH_DIST = 0.7f;
const float Tracker::TH_MAX_RATIO_FH = 0.03f;
const float Tracker::TH_COS_PARALLAX = 0.9999f;
const float Tracker::TH_POSE_SEL = 0.7f;
const float Tracker::TH_MIN_RATIO_TRIANG_PTS = 0.9f;

Tracker::Tracker(System::Mode eMode, const std::shared_ptr<Map>& pMap) :
    meMode(eMode), mState(State::NOT_INITIALIZED), mbFirstFrame(true),
    mpView1(nullptr), mpView2(nullptr), mpMap(pMap)
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

void Tracker::trackImgsMono(const cv::Mat& img, double timestamp)
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
        setAbsPose(pFrameCur->mPose);
    } else {
        mpView1 = mvpFramesPrev[0] = mvpFramesCur[0];
        mpView2 = mvpFramesCur[0] = pFrameCur;
        if (mState == State::NOT_INITIALIZED) {
            mState = initializeMap();
        } else if (mState == State::OK) {
            // TODO: tracking scheme after map is initialized
            mState = State::NOT_INITIALIZED;
        } else if (mState == State::LOST) {
            // TODO: relocalization?
            mState = State::NOT_INITIALIZED;
        }
    }
    imgPrev = imgCur;
}

cv::Mat Tracker::rgb2Gray(const cv::Mat& img) const
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

Tracker::State Tracker::initializeMap()
{
    if (meMode == System::Mode::MONOCULAR) {
        return initializeMapMono();
    }
    return State::NOT_INITIALIZED;
}

Tracker::State Tracker::initializeMapMono()
{
    // match features between previous (1) and current (2) frame
    vector<cv::DMatch> vMatches = matchFeatures2Dto2D();
    // temp test on display of feature matching result
    displayFeatMatchResult(vMatches, 0, 0);
    // compute fundamental matrix F (from previous (1) to current (2))
    Mat F21 = computeFundamental(vMatches);
    // compute homography H (from previous (1) to current (2))
    Mat H21 = computeHomography(vMatches);
    //cout << "F21 = " << endl << F21 << endl << "H21 = " << endl << H21 << endl;
    // recover pose from F or H
    Mat Xw3Ds;
    vector<int> vIdxPts;
    if (recoverPoseFromFH(vMatches, F21, H21, Xw3Ds, vIdxPts)) {
        cout << "Number of triangulated points (good/total) = "
             << vIdxPts.size() << "/" << Xw3Ds.cols << endl;
        // TODO: initialize 3D map points by triangulating 2D keypoints
        buildInitialMap(Xw3Ds, vMatches, vIdxPts);
        cv::waitKey();
    }
    // temp: always return false as the initalization scheme is incomplete
    return State::NOT_INITIALIZED; 
}

std::vector<cv::DMatch> Tracker::matchFeatures2Dto2D() const
{
    vector<cv::DMatch> vMatches;
    vector<vector<cv::DMatch>> vknnMatches;
    mpFeatMatcher->knnMatch(mpView1->getFeatDescriptors(), // query
                            mpView2->getFeatDescriptors(), // train
                            vknnMatches, 2); // get 2 best matches
    // find good matches using Lowe's ratio test
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    vMatches.reserve(vknnMatches.size());
    for (unsigned i = 0; i < vknnMatches.size(); ++i) {
        if (vknnMatches[i][0].distance < TH_DIST * vknnMatches[i][1].distance) {
            // filter out-of-border matches
            const cv::Point2f& pt1 = vKpts1[vknnMatches[i][0].queryIdx].pt;
            const cv::Point2f& pt2 = vKpts2[vknnMatches[i][0].trainIdx].pt;
            if (is2DPtInBorder(Mat(pt1)) && is2DPtInBorder(Mat(pt2))) {
                vMatches.push_back(vknnMatches[i][0]);
            }
        }
    }
    return vMatches;
}

inline bool Tracker::is2DPtInBorder(const cv::Mat& pt) const
{
    bool result =
        (pt.at<float>(0) >= 0 && pt.at<float>(0) < Config::width()) &&
        (pt.at<float>(1) >= 0 && pt.at<float>(1) < Config::height());
    return result;
}

void Tracker::displayFeatMatchResult(const std::vector<cv::DMatch>& vMatches,
                                     int viewPrev, int viewCur) const
{
    const Mat& imgPrev = mvImgsPrev[viewPrev];
    const Mat& imgCur = mvImgsCur[viewCur];
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
    //cv::waitKey();
}

Mat Tracker::computeFundamental(const std::vector<cv::DMatch>& vMatches) const
{
    // fundamental matrix result (from previous (1) to current (2))
    Mat F21;
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    unsigned nMatches = vMatches.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPts1, vPts2; 
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[vMatches[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[vMatches[i].trainIdx];
        vPts1.push_back(kpt1.pt);
        vPts2.push_back(kpt2.pt);
    }
    // F computation (using RANSAC)
    F21 = cv::findFundamentalMat(vPts1, vPts2, cv::FM_RANSAC, 1.0, 0.99,
                                 cv::noArray());
    F21.convertTo(F21, CV_32FC1); // maintain precision
    return F21;
}

cv::Mat Tracker::computeHomography(
    const std::vector<cv::DMatch>& vMatches) const
{
    // fundamental matrix result (from previous (1) to current (2))
    Mat H21;
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    unsigned nMatches = vMatches.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPts1, vPts2; 
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[vMatches[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[vMatches[i].trainIdx];
        vPts1.push_back(kpt1.pt);
        vPts2.push_back(kpt2.pt);
    }
    // H computation (using RANSAC)
    H21 = cv::findHomography(vPts1, vPts2, cv::RANSAC, 1., cv::noArray(),
                             2000, 0.99);
    H21.convertTo(H21, CV_32FC1);
    return H21;
}

bool Tracker::recoverPoseFromFH(const std::vector<cv::DMatch>& vMatches,
                                const cv::Mat& F21,
                                const cv::Mat& H21,
                                cv::Mat& Xw3Ds,
                                std::vector<int>& vIdxGoodPts)
{
    // select the better transformation from either F or H
    FHResult resFH = selectFH(vMatches, F21, H21);
    const Mat K = Config::K(); // cam intrinsics
    vector<Mat> vR21s; // possible rotations
    vector<Mat> vt21s; // possible translations
    int nSolutions = 0; // number of possible poses
    // decompose F or H into [R|t]s
    //if (resFH == FHResult::F) {
    //    // 4 solutions
    //    nSolutions = decomposeFforRT(F21, vR21s, vt21s);
    //} else if (resFH == FHResult::H) {
    //    vector<Mat> vNormal21s; // plane normal matrices
    //    // at most 8 solutions
    //    nSolutions = decomposeHforRT(H21, vR21s, vt21s, vNormal21s);
    //} else if (resFH == FHResult::NONE) {
    //    return false;
    //}

    // temp test 20190828: traverse all [R|t] solutions
    vector<Mat> vR21sF, vt21sF, vR21sH, vt21sH, vNormal21sH;
    int nSolF = decomposeFforRT(F21, vR21sF, vt21sF);
    int nSolH = decomposeHforRT(H21, vR21sH, vt21sH, vNormal21sH);
    vR21s.reserve(nSolF + nSolH);
    vt21s.reserve(nSolF + nSolH);
    for (int i = 0; i < nSolF; ++i) {
        vR21s.push_back(vR21sF[i]);
        vt21s.push_back(vt21sF[i]);
    }
    for (int i = 0; i < nSolH; ++i) {
        vR21s.push_back(vR21sH[i]);
        vt21s.push_back(vt21sH[i]);
    }
    nSolutions = nSolF + nSolH;
  
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    // get the vector of 2D keypoints from both frames
    unsigned nMatches = vMatches.size();
    vector<cv::Point2f> vPts1;
    vector<cv::Point2f> vPts2;
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[vMatches[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[vMatches[i].trainIdx];
        vPts1.push_back(kpt1.pt);
        vPts2.push_back(kpt2.pt);
    }
    // get best pose & corresponding triangulated 3D points from the solutions
    vector<int> vnGoodPts(nSolutions, 0);
    vector<Mat> vXws(nSolutions); // triangulated 3D points
    // indices of good triangulated points for all solutions
    vector<vector<int>> vvIdxPts(nSolutions);
    int nMaxGoodPts = 0;
    int nIdxBestPose = -1;
    for (int i = 0; i < nSolutions; ++i) {
        vvIdxPts[i].reserve(nMatches);
        // maintain precision
        vR21s[i].convertTo(vR21s[i], CV_32FC1);
        vt21s[i].convertTo(vt21s[i], CV_32FC1);
        // pose T = K[I|0] for previous frame
        Mat Tcw1 = Mat::zeros(3, 4, CV_32FC1);
        K.copyTo(Tcw1.rowRange(0, 3).colRange(0, 3));
        // pose T = K[R|t] for current frame
        Mat Tcw2 = Mat::zeros(3, 4, CV_32FC1);
        vR21s[i].copyTo(Tcw2.rowRange(0, 3).colRange(0, 3));
        vt21s[i].copyTo(Tcw2.rowRange(0, 3).col(3));
        Tcw2 = K * Tcw2;
        // compute triangulated 3D world points
        Mat Xw4Ds;
        cv::triangulatePoints(Tcw1, Tcw2, vPts1, vPts2, Xw4Ds);
        Xw4Ds.convertTo(Xw4Ds, CV_32FC1);
        Mat Xws = Xw4Ds.rowRange(0, 3);
        for (int j = 0; j < Xw4Ds.cols; ++j) {
            Xws.col(j) /= Xw4Ds.at<float>(3, j);
        }
        vXws[i] = Xws;
        // check number of good points
        CamPose tmpPose = CamPose(vR21s[i], vt21s[i]);
        for (unsigned j = 0; j < nMatches; ++j) {
            Mat Xw = Xws.col(j);
            const cv::KeyPoint& kpt1 = vKpts1[vMatches[j].queryIdx];
            const cv::KeyPoint& kpt2 = vKpts2[vMatches[j].trainIdx];
            if (checkTriangulatedPt(Xw, kpt1, kpt2, tmpPose)) {
                vvIdxPts[i].push_back(j);
                vnGoodPts[i]++;
            }
        }
        if (vnGoodPts[i] > nMaxGoodPts) {
            nMaxGoodPts = vnGoodPts[i];
            nIdxBestPose = i;
        }
    }

    // temp test 20190828: traverse all [R|t] solutions
    resFH = nIdxBestPose >= nSolF ? FHResult::H : FHResult::F;
    
    // select the most possible pose
    int nBestPose = 0;
    for (const int& nGoodPt : vnGoodPts) {
        if (nGoodPt > TH_MIN_RATIO_TRIANG_PTS * nMatches &&
            nGoodPt > TH_POSE_SEL * nMaxGoodPts) {
            nBestPose++;
        }
    }
    if (nBestPose == 1) {
        // store pose & triangulated points
        mpView2->mPose.setPose(vR21s[nIdxBestPose], vt21s[nIdxBestPose]);
        vXws[nIdxBestPose].copyTo(Xw3Ds);
        vIdxGoodPts = vvIdxPts[nIdxBestPose];

        // temp display
        for (unsigned i = 0; i < vnGoodPts.size(); ++i) {
            cout << "solution " << i << ": " << vnGoodPts[i] << endl;
        }
        unsigned nCurFrmIdx = mpView2->getFrameIdx();
        cout << "Pose T_{" << nCurFrmIdx << "|" << nCurFrmIdx - 1
             <<"} recovered from " << (resFH == FHResult::F ? "F" : "H")
             << endl;// << mpView2->mPose.getPose() << endl;
        cout << mpView2->mPose << endl;
        
        return true;
    }
    return false;
}

Tracker::FHResult Tracker::selectFH(const std::vector<cv::DMatch>& vMatches,
                                    const cv::Mat& F21,
                                    const cv::Mat& H21) const
{
    // compute reprojection errors for F & H result:
    // error_F = (1/N) * sigma_i(d(x_{2,i}, F_{21} x_{1,i})^2 +
    //                           d(x_{1,i}, F_{21}^T x_{2,i})^2
    // note: error_F is the mean distance between point and epipolar line
    // error_H = (1/N) * sigma_i(d(x_{1,i}, H_{21} x_{1,i})^2 +
    //                           d(x_{2,i}, H21^{-1} x_{2,i})^2)
    Mat F12, H12; // inverse matrix of H and F
    F12 = F21.t();
    H12 = H21.inv(cv::DECOMP_LU);
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    // compute reprojection errors
    float errorF = 0.f;
    float errorH = 0.f;
    unsigned nMatches = vMatches.size();
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[vMatches[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[vMatches[i].trainIdx];
        // use homogeneous coordinate
        Mat x1 = (cv::Mat_<float>(3, 1) << kpt1.pt.x, kpt1.pt.y, 1.f);
        Mat x2 = (cv::Mat_<float>(3, 1) << kpt2.pt.x, kpt2.pt.y, 1.f);
        errorF += computeReprojErr(F21, F12, x1, x2, ReprojErrScheme::F);
        errorH += computeReprojErr(H21, H12, x1, x2, ReprojErrScheme::H);
    }
    errorF /= static_cast<float>(nMatches);
    errorH /= static_cast<float>(nMatches);
    cout << "eF = " << errorF << " | eH = " << errorH << " -> ";
    // similarity between error of F and H
    float errFHRatio = errorF / errorH;
    if (errFHRatio > TH_MAX_RATIO_FH) {
        cout << "H" << endl;
        return FHResult::H;
    } else if (errFHRatio <= TH_MAX_RATIO_FH) {
        cout << "F" << endl;
        return FHResult::F;
    }
    // no result if errors are similar between F and H
    cout << "None" << endl;
    return FHResult::NONE;
}

int Tracker::decomposeFforRT(const cv::Mat& F21,
                             std::vector<cv::Mat>& vR21s,
                             std::vector<cv::Mat>& vt21s) const
{
    // ref: Richard Hartley and Andrew Zisserman. Multiple view geometry
    // in computer vision. Cambridge university press, 2003.
    // 4 possibilities
    vR21s.resize(4);
    vt21s.resize(4);
    // E_{21} = K_2^T  F_{21} K_1 (K_2 == K_1 == K)
    Mat K = Config::K();
    Mat E21 = K.t() * F21 * K;
    Mat R211, R212, t21; // 2 R21s and 1 t21 result
    cv::decomposeEssentialMat(E21, R211, R212, t21);
    // solution 1: [R_1|t]
    vR21s[0] = R211.clone();
    vt21s[0] = t21.clone();
    // solution 2: [R_1|-t]
    vR21s[1] = R211.clone();
    vt21s[1] = -t21.clone();
    // solution 3: [R_2|t]
    vR21s[2] = R212.clone();
    vt21s[2] = t21.clone();
    // solution 4: [R_2|-t]
    vR21s[3] = R212.clone();
    vt21s[3] = -t21.clone();
    return 4; // 4 possible solutions
}

int Tracker::decomposeHforRT(const cv::Mat& H21,
                             std::vector<cv::Mat>& vR21s,
                             std::vector<cv::Mat>& vt21s,
                             std::vector<cv::Mat>& vNormal21s) const
{
    // ref: Ezio Malis, Manuel Vargas, and others.
    // Deeper understanding of the homography decomposition for
    // vision-based control. 2007.
    // at most 8 possibilities
    vR21s.reserve(8);
    vt21s.reserve(8);
    vNormal21s.reserve(8);
    // return number of possibilities of [R|t]
    return cv::decomposeHomographyMat(H21, Config::K(), vR21s, vt21s,
                                      vNormal21s);
}

float Tracker::computeReprojErr(const cv::Mat& T21, const cv::Mat& T12,
                                const cv::Mat& p1, const cv::Mat& p2,
                                ReprojErrScheme scheme) const
{
    // compute reprojection errors for F & H result:
    // error_F = d(x_{2,i}, F_{21} x_{1,i})^2 + d(x_{1,i}, F_{21}^T x_{2,i})^2
    // error_H = d(x_{1,i}, H_{21} x_{1,i})^2 + d(x_{2,i}, H21^{-1} x_{2,i})^2
    float err = 0.0f;
    if (scheme == ReprojErrScheme::F) {
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
    } else if (scheme == ReprojErrScheme::H) {
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
        assert(0); // TODO: add other schemes
    }
    return err;
}

bool Tracker::checkTriangulatedPt(const cv::Mat& Xw,
                                  const cv::KeyPoint& kpt1,
                                  const cv::KeyPoint& kpt2,
                                  const CamPose& pose) const
{
    // world coord in current frame / cam coord in previous frame
    const Mat& Xc1 = Xw;
    // 3D cam coord in Current frame
    Mat Rcw = pose.getRotation();
    Mat tcw = pose.getTranslation();
    Mat Xc2 = Rcw*Xc1 + tcw;
    // condition 1: must be positive depth in both views
    float invDepth1 = 1.f / Xc1.at<float>(2);
    float invDepth2 = 1.f / Xc2.at<float>(2);
    if (invDepth1 < 0 || invDepth2 < 0) { 
        return false;
    }
    // condition 2: the parallax of 2 views must not be too small
    Mat O1 = Mat::zeros(3, 1, CV_32FC1); // 3D cam origin in previous frame
    Mat O2 = pose.getCamOrigin(); // 3D cam origin in current frame
    Mat Xwo1 = O1 - Xc1; // vector from Xw to o1
    Mat Xwo2 = O2 - Xc1; // vector from Xw to o2
    float normXwo1 = cv::norm(Xwo1, cv::NORM_L2);
    float normXwo2 = cv::norm(Xwo2, cv::NORM_L2);
    float cosParallax = Xwo1.dot(Xwo2) / (normXwo1 * normXwo2);
    if (cosParallax > TH_COS_PARALLAX) {
        return false;
    }
    // condition 3: reprojected 2D point needs to be inside image border
    const Mat K = Config::K();
    // projected cam coords in previous and current frames
    Mat Xc1Proj = invDepth1 * K * Xc1; 
    Mat Xc21roj = invDepth2 * K * Xc2;
    // reprojected 2D image coords in both frames
    Mat x1Reproj = Xc1Proj.rowRange(0, 2);
    Mat x2Reproj = Xc21roj.rowRange(0, 2);
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
    return true; // triangulated result is good if all conditions are met
}

void Tracker::buildInitialMap(const cv::Mat& Xws,
                              const std::vector<cv::DMatch>& vMatches,
                              const std::vector<int> vIdxPts) const
{
    // clear the map to build a new one
    mpMap->clear();
    // traverse all triangulated points
    int frmIdx2 = mpView2->getFrameIdx();
    Mat descs2 = mpView2->getFeatDescriptors();
    for (unsigned i = 0; i < vIdxPts.size(); ++i) {
        int nIdxDesc2 = vMatches[vIdxPts[i]].trainIdx;
        Mat Xw = Xws.col(i); // i-th column for i-th point
        Mat desc2 = descs2.row(nIdxDesc2); // i-th row for i-th descriptor
        // add point into the map
        shared_ptr<MapPoint> pX =
            make_shared<MapPoint>(MapPoint(Xw, desc2, frmIdx2));
        mpMap->addPt(pX);
    }
}

} // Namespace SLAM_demo
