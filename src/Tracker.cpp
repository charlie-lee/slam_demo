/**
 * @file   Tracker.cpp
 * @brief  Implementations of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#include "Tracker.hpp"

#include <cmath>
#include <iostream> // temp for debugging
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <opencv2/calib3d.hpp> // cv::undistort(), H & F computation, ...
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // cv::imshow()
#include <opencv2/imgproc.hpp>
//#include <Eigen/Core>
#include "CamPose.hpp"
#include "Config.hpp"
#include "FeatureMatcher.hpp"
#include "Frame.hpp"
#include "KeyFrame.hpp"
#include "LocalMapper.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"
#include "Optimizer.hpp"
#include "Utility.hpp"

namespace SLAM_demo {

using std::map;
using std::shared_ptr;
using std::make_shared;
using std::set;
using std::vector;
using cv::Mat;

using std::cout;
using std::endl;

// constants
const float Tracker::TH_MAX_RATIO_FH = 0.2f;
const float Tracker::TH_REPROJ_ERR_FACTOR = 3.0f;
const float Tracker::TH_POSE_SEL = 0.8f;
const float Tracker::TH_MIN_RATIO_TRIANG_PTS = 0.5f;
const int Tracker::TH_MIN_MATCHES_2D_TO_3D = 50;
// other data
unsigned Tracker::n1stFrame = 0;

Tracker::Tracker(System::Mode eMode, const std::shared_ptr<Map>& pMap,
                 const std::shared_ptr<Optimizer>& pOptimizer,
                 const std::shared_ptr<LocalMapper>& pLocalMapper) :
    meMode(eMode), meState(State::NOT_INITIALIZED), mbFirstFrame(true),
    mpMap(pMap), mpOpt(pOptimizer), mpLocalMapper(pLocalMapper),
    mpView1(nullptr), mpView2(nullptr), mpKFLatest(nullptr)
{
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
    // RGB -> Grayscale
    Mat& imgPrev = mvImgsPrev[0];
    Mat& imgCur = mvImgsCur[0];
    imgCur = img;
    Mat imgCurGray = rgb2Gray(img);
    // initialize each frame
    shared_ptr<Frame> pFrameCur =
        make_shared<Frame>(imgCurGray, timestamp);

    cout << "keypoints | map points | keyframes: "
         << pFrameCur->keypoints().size() << " | "
         << mpMap->getAllMPts().size() << " | "
         << mpMap->getAllKFs().size() << endl;
    
    // feature matching
    if (mbFirstFrame) {
        mvpFramesCur[0] = pFrameCur;
        mbFirstFrame = false;
        meState = State::NOT_INITIALIZED;
    } else {
        mpView1 = mvpFramesPrev[0] = mvpFramesCur[0];
        mpView2 = mvpFramesCur[0] = pFrameCur;
        if (meState == State::NOT_INITIALIZED) {
            meState = initializeMap();
        } else if (meState == State::OK) {
            meState = track();
        } else if (meState == State::LOST) {
            meState = track();
            System::nLostFrames++;
        }
    }
    imgPrev = imgCur;
}

cv::Mat Tracker::rgb2Gray(const cv::Mat& img) const
{
    Mat imgGray;
    if (img.channels() == 3) {
        cv::cvtColor(img, imgGray, cv::COLOR_RGB2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, imgGray, cv::COLOR_RGBA2GRAY);
    } else {
        imgGray = img.clone();
    }
    return imgGray;
}

Tracker::State Tracker::initializeMap()
{
    // clear all poses and add an initial pose T_{k|k} = [I|0]
    // where k is the 1st frame in a initialized SLAM system.
    mvTs.clear();
    setAbsPose(CamPose());
    Tracker::n1stFrame = mpView1->index();
    if (meMode == System::Mode::MONOCULAR) {
        return initializeMapMono();
    }
    return State::NOT_INITIALIZED;
}

Tracker::State Tracker::initializeMapMono()
{
    // match features between previous (1) and current (2) frame
    shared_ptr<FeatureMatcher> pFMatcher = make_shared<FeatureMatcher>(
        //50.0f, false); //true, 0.75f);
        50.0f, true, 0.7f, 30, 64);
    //mvMatches2Dto2D = pFMatcher->match2Dto2D(mpView2, mpView1);
    mvMatches2Dto2D = pFMatcher->match2Dto2DCustom(mpView2, mpView1);

    // temp test on display of feature matching result
    displayFeatMatchResult(0, 0);

    // compute fundamental matrix F (from previous (1) to current (2))
    Mat F21 = computeFundamental();
    // compute homography H (from previous (1) to current (2))
    Mat H21 = computeHomography();
    //cout << "F21 = " << endl << F21 << endl << "H21 = " << endl << H21 << endl;
    // recover pose from F or H
    Mat Xw3Ds;
    vector<int> vnIdxPts;
    if (recoverPoseFromFH(F21, H21, Xw3Ds, vnIdxPts)) {
        mvTs.push_back(mpView2->mPose); // get pose T_{k+1|k}
        
        cout << "Number of triangulated points (good/total) = "
             << vnIdxPts.size() << "/" << Xw3Ds.cols << endl;

        buildInitialMap(Xw3Ds, vnIdxPts);
        
        // optimize both pose & map data
        mpOpt->globalBundleAdjustment(0, 20, true);

        // assign velocity
        mVelocity = mpView2->mPose;
 
        return State::OK;
    }
    return State::NOT_INITIALIZED; 
}

void Tracker::displayFeatMatchResult(int viewPrev, int viewCur) const
{
    // get keypoint data
    vector<cv::KeyPoint> vKpts1;
    vector<cv::KeyPoint> vKpts2 = mvpFramesCur[viewCur]->keypoints();
    const vector<cv::DMatch>* pvMatches = nullptr;
    if (meState == State::NOT_INITIALIZED) {
        vKpts1 = mvpFramesPrev[viewPrev]->keypoints();
        pvMatches = &mvMatches2Dto2D;
    } else if (meState == State::OK || meState == State::LOST) {
        // Construct a vector of fake keypoints for the map points
        vKpts1.resize(mvpMPts.size());
        //vKpts1.resize(mvpFramesPrev[viewPrev]->keypoints().size());
        int nMatches = mvMatches2Dto3D.size();
        // get reprojected keypoint positions based on view 1 (viewPrev)
        for (int i = 0; i < nMatches; ++i) {
            int nIdxMPt = mvMatches2Dto3D[i].trainIdx;
            shared_ptr<MapPoint> pMPt = mvpMPts[nIdxMPt];
            //shared_ptr<MapPoint> pMPt =
            //    mvpFramesPrev[viewPrev]->mappoint(nIdxMPt);
            // reprojected image coord in view 1
            if (pMPt) {
                Mat x1 = mvpFramesPrev[viewPrev]->coordWorld2Img(
                    pMPt->X3D());
                vKpts1[nIdxMPt] = cv::KeyPoint(
                    x1.at<float>(0), x1.at<float>(1), 1 /* size */);
            } else {
                vKpts1[nIdxMPt] = cv::KeyPoint(-1, -1, 0);
            }
        }
        pvMatches = &mvMatches2Dto3D;
    }
    // undistort input images
    const Mat& imgPrev = mvImgsPrev[viewPrev];
    const Mat& imgCur = mvImgsCur[viewCur];
    Mat imgPrevUnD, imgCurUnD, imgOut;
    cv::undistort(imgPrev, imgPrevUnD, Config::K(), Config::distCoeffs());
    cv::undistort(imgCur, imgCurUnD, Config::K(), Config::distCoeffs());
    
    // display keypoint matches (undistorted) on undistorted images
    //cv::drawMatches(
    //    imgPrevUnD, vKpts1, imgCurUnD, vKpts2, *pvMatches, imgOut,
    //    cv::Scalar({255, 0, 0}), // color for matching line (BGR)
    //    cv::Scalar({0, 255, 0}), // color for keypoint (BGR)
    //    vector<char>(), // mask
    //    cv::DrawMatchesFlags::DEFAULT);
    
    // display matching result on current view (undistorted) only
    // force output image to be colored
    imgOut = imgCur.clone();
    if (imgOut.channels() == 1) {
        cv::cvtColor(imgOut, imgOut, cv::COLOR_GRAY2BGR);
    }
    // draw markers on keypoints
    //for (const auto& kpt : vKpts1) {
    //    if (Utility::is2DPtInBorder(Mat(kpt.pt))) {
    //        cv::drawMarker(imgOut, kpt.pt,
    //                       cv::Scalar(127, 127, 0), // marker color
    //                       cv::MARKER_DIAMOND, // marker shape
    //                       5); // marker size
    //    }
    //}
    for (const auto& kpt : vKpts2) {
        if (Utility::is2DPtInBorder(Mat(kpt.pt))) {
            cv::drawMarker(imgOut, kpt.pt, cv::Scalar(0, 255, 0),
                           cv::MARKER_SQUARE, 5);
        }
    }
    // draw arrowed lines from keypoints in view 1 to view 2 on matches
    for (unsigned i = 0; i < pvMatches->size(); ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[(*pvMatches)[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[(*pvMatches)[i].queryIdx];
        cv::arrowedLine(imgOut, kpt1.pt, kpt2.pt,
                        cv::Scalar(0, 0, 255), // line color
                        1 /* line thickness */, cv::LINE_AA,
                        0 /* shift */, 0.2 /* arrowtip size ratio*/);
    }
    
    // display matching result
    cv::String winName("Feature matches between previous and current frame");
    cv::namedWindow(winName, cv::WINDOW_KEEPRATIO);
    cv::imshow(winName, imgOut);
}

cv::Mat Tracker::computeFundamental() const
{
    // fundamental matrix result (from previous (1) to current (2) frame)
    Mat F21;
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->keypoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->keypoints();
    unsigned nMatches = mvMatches2Dto2D.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPts1, vPts2; 
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].queryIdx];
        vPts1.push_back(kpt1.pt);
        vPts2.push_back(kpt2.pt);
    }
    // F computation (using RANSAC)
    F21 = cv::findFundamentalMat(vPts1, vPts2, cv::FM_RANSAC, 1.0, 0.99,
                                 cv::noArray());
    F21.convertTo(F21, CV_32FC1); // maintain precision
    return F21;
}

cv::Mat Tracker::computeHomography() const
{
    // fundamental matrix result (from previous (1) to current (2))
    Mat H21;
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->keypoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->keypoints();
    unsigned nMatches = mvMatches2Dto2D.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPts1, vPts2; 
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].queryIdx];
        vPts1.push_back(kpt1.pt);
        vPts2.push_back(kpt2.pt);
    }
    // H computation (using RANSAC)
    H21 = cv::findHomography(vPts1, vPts2, cv::RANSAC, 1., cv::noArray(),
                             2000, 0.99);
    H21.convertTo(H21, CV_32FC1);
    return H21;
}

bool Tracker::recoverPoseFromFH(const cv::Mat& F21, const cv::Mat& H21,
                                cv::Mat& Xw3Ds, std::vector<int>& vnIdxGoodPts)
{
    // select the better transformation from either F or H
    FHResult eResFH = selectFH(F21, H21);
    const Mat K = Config::K(); // cam intrinsics
    vector<Mat> vR21s; // possible rotations
    vector<Mat> vt21s; // possible translations
    int nSolutions = 0; // number of possible poses
    // decompose F or H into [R|t]s
    if (eResFH == FHResult::F) {
        // 4 solutions
        nSolutions = decomposeFforRT(F21, vR21s, vt21s);
    } else if (eResFH == FHResult::H) {
        vector<Mat> vNormal21s; // plane normal matrices
        // at most 8 solutions
        nSolutions = decomposeHforRT(H21, vR21s, vt21s, vNormal21s);
    } else if (eResFH == FHResult::NONE) {
        return false;
    }

    // temp test 20190828: traverse all [R|t] solutions
    //vector<Mat> vR21sF, vt21sF, vR21sH, vt21sH, vNormal21sH;
    //int nSolF = decomposeFforRT(F21, vR21sF, vt21sF);
    //int nSolH = decomposeHforRT(H21, vR21sH, vt21sH, vNormal21sH);
    //vR21s.reserve(nSolF + nSolH);
    //vt21s.reserve(nSolF + nSolH);
    //for (int i = 0; i < nSolF; ++i) {
    //    vR21s.push_back(vR21sF[i]);
    //    vt21s.push_back(vt21sF[i]);
    //}
    //for (int i = 0; i < nSolH; ++i) {
    //    vR21s.push_back(vR21sH[i]);
    //    vt21s.push_back(vt21sH[i]);
    //}
    //nSolutions = nSolF + nSolH;
  
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->keypoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->keypoints();
    // get the vector of 2D keypoints from both frames
    unsigned nMatches = mvMatches2Dto2D.size();
    vector<cv::Point2f> vPts1;
    vector<cv::Point2f> vPts2;
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].queryIdx];
        vPts1.push_back(kpt1.pt);
        vPts2.push_back(kpt2.pt);
    }
    // get best pose & corresponding triangulated 3D points from the solutions
    vector<int> vnGoodPts(nSolutions, 0);
    vector<Mat> vXws(nSolutions); // triangulated 3D points
    // indices of good triangulated points for all solutions
    vector<vector<int>> vvnIdxPts(nSolutions);
    int nMaxGoodPts = 0;
    int nIdxBestPose = -1;
    for (int i = 0; i < nSolutions; ++i) {
        vvnIdxPts[i].reserve(nMatches);
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
            const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[j].trainIdx];
            const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[j].queryIdx];
            if (Utility::checkTriangulatedPt(Xw, kpt1, kpt2,
                                             mpView1->mPose, tmpPose)) {
                vvnIdxPts[i].push_back(j);
                vnGoodPts[i]++;
            }
        }
        if (vnGoodPts[i] > nMaxGoodPts) {
            nMaxGoodPts = vnGoodPts[i];
            nIdxBestPose = i;
        }
    }

    // temp test 20190828: traverse all [R|t] solutions
    //eResFH = nIdxBestPose >= nSolF ? FHResult::H : FHResult::F;
    
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
        vnIdxGoodPts = vvnIdxPts[nIdxBestPose];

        // temp display
        for (unsigned i = 0; i < vnGoodPts.size(); ++i) {
            cout << "solution " << i << ": " << vnGoodPts[i] << endl;
        }
        cout << "Pose T_{" << Tracker::n1stFrame + 1
             << "|" << Tracker::n1stFrame
             <<"} recovered from " << (eResFH == FHResult::F ? "F" : "H")
             << endl;// << mpView2->mPose.getPose() << endl;
        cout << mpView2->mPose << endl;

        // check reprojection error
        float errorRT = 0.0f;
        for (unsigned i = 0; i < vnIdxGoodPts.size(); ++i) {
            int nIdxX = vnIdxGoodPts[i];
            int nIdxx = nIdxX;
            Mat xReproj = mpView2->coordWorld2Img(Xw3Ds.col(nIdxX));
            Mat x = Mat(vPts2[nIdxx]);
            Mat diffx = xReproj - x;
            errorRT += diffx.dot(diffx);
        }
        errorRT /= vnIdxGoodPts.size();
        cout << "Mean square reprojection error for init = " << errorRT << endl;
        
        return true;
    }
    return false;
}

Tracker::FHResult Tracker::selectFH(const cv::Mat& F21,
                                    const cv::Mat& H21) const
{
    // compute reprojection errors for F & H result:
    // error_F = (1/N) * sigma_i(d(x_{2,i}, F_{21} x_{1,i})^2 +
    //                           d(x_{1,i}, F_{21}^T x_{2,i})^2
    // note: error_F is the mean distance between point and epipolar line
    // error_H = (1/N) * sigma_i(d(x_{1,i}, H_{21} x_{1,i})^2 +
    //                           d(x_{2,i}, H21^{-1} x_{2,i})^2)
    if (F21.empty() && H21.empty()) {
        return FHResult::NONE;
    } else if (F21.empty()) {
        return FHResult::H;
    } else if (H21.empty()) {
        return FHResult::F;
    }
    
    Mat F12, H12; // inverse matrix of H and F
    F12 = F21.t();
    H12 = H21.inv(cv::DECOMP_LU);
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->keypoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->keypoints();
    // compute reprojection errors
    float errorF = 0.f;
    float errorH = 0.f;
    unsigned nMatches = mvMatches2Dto2D.size();
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].trainIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].queryIdx];
        // use homogeneous coordinate
        Mat x1 = (cv::Mat_<float>(3, 1) << kpt1.pt.x, kpt1.pt.y, 1.f);
        Mat x2 = (cv::Mat_<float>(3, 1) << kpt2.pt.x, kpt2.pt.y, 1.f);
        errorF += Utility::computeReprojErr(F21, F12, x1, x2,
                                            Utility::ReprojErrScheme::F);
        errorH += Utility::computeReprojErr(H21, H12, x1, x2,
                                            Utility::ReprojErrScheme::H);
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

void Tracker::buildInitialMap(const cv::Mat& Xws,
                              const std::vector<int>& vnIdxPts)
{
    // clear the map to build a new one
    mpMap->clear();
    // create new keyframe
    shared_ptr<KeyFrame> pKF1 = make_shared<KeyFrame>(mpView1);
    shared_ptr<KeyFrame> pKF2 = make_shared<KeyFrame>(mpView2);
    // traverse all triangulated points
    for (unsigned i = 0; i < vnIdxPts.size(); ++i) {
        int nIdxKpt1 = mvMatches2Dto2D[vnIdxPts[i]].trainIdx;
        int nIdxKpt2 = mvMatches2Dto2D[vnIdxPts[i]].queryIdx;
        // Xws have all triangulated points, here only select the valid ones
        Mat Xw = Xws.col(vnIdxPts[i]); // i-th column for i-th point
        shared_ptr<MapPoint> pMPt = make_shared<MapPoint>(mpMap, Xw);
        // add observations for map point & frame
        pMPt->addObservation(pKF1, nIdxKpt1);
        pMPt->addObservation(pKF2, nIdxKpt2);
        pKF1->bindMPt(pMPt, nIdxKpt1);
        pKF2->bindMPt(pMPt, nIdxKpt2);
        // update map point data for current frame for tracking next frame
        mpView2->bindMPt(pMPt, nIdxKpt2);
        // update distinctive descriptor
        pMPt->updateDescriptor();
        // add point into the map & update data of related frames
        mpMap->addMPt(pMPt);
    }
    // update covisibility graph
    pKF1->updateConnections();
    pKF2->updateConnections();
    // set latest added keyframe
    mpKFLatest = pKF2;
}

Tracker::State Tracker::track()
{
    State eState = State::NOT_INITIALIZED;
        
    // match features between previous (1) and current (2) frame
    shared_ptr<FeatureMatcher> pFMatcher = make_shared<FeatureMatcher>(
        64.0f, false); //true, 0.75f);
    //64.0f, true, 1.0f, 45, 64);
    mvMatches2Dto3D = pFMatcher->match2Dto3D(mpView2, mpView1);
    //mvMatches2Dto3D = pFMatcher->match2Dto3DCustom(mpView2, mpView1);

    int nInliers;
    // pose estimation
    if (mvMatches2Dto3D.size() < 100) {
        nInliers = poseEstimation(mVelocity * mpView1->mPose);
        cout << "RANSAC PnP: "
             << nInliers << "/" << mvMatches2Dto3D.size() << "; ";
    } else {
        mpView2->mPose = mVelocity * mpView1->mPose;
    }
    
    // only optimize pose
    nInliers = mpOpt->poseOptimization(mpView2);
    cout << "pose BA: " << nInliers << "/"
         << mvMatches2Dto3D.size();

    // get all related map points
    mvpMPts = trackLocalMap();
    pFMatcher = make_shared<FeatureMatcher>(4.0f, true, 1.0f, 15, 100);
    mvMatches2Dto3D = pFMatcher->match2Dto3DCustom(mpView2, mvpMPts);
    if (mvMatches2Dto3D.size() < TH_MIN_MATCHES_2D_TO_3D) {
        mvpMPts = mpMap->getAllMPts();
        pFMatcher = make_shared<FeatureMatcher>(8.0f, true, 0.9f, 45, 100);
        //mvMatches2Dto3D = pFMatcher->match2Dto3D(mpView2, mvpMPts);
        mvMatches2Dto3D = pFMatcher->match2Dto3DCustom(mpView2, mvpMPts);
        // Try to find a better initial pose by PnP
        poseEstimation(mpView1->mPose);
    }
    
    nInliers = mpOpt->poseOptimization(mpView2);
    cout << " -> " << nInliers << "/"
         << mpView2->getNumMPts() << endl;

    // update trackability data for all tracked map points
    updateMPtTrackedData();

    // temp test on display of feature matching result
    displayFeatMatchResult(0, 0);

    if (qualifiedAsKeyFrame(nInliers)) {
        addNewKeyFrame();
    }
    
    if (nInliers > 10) {
        cout << "Pose T_{" << mpView2->index() << "|"
             << Tracker::n1stFrame  << "}: "
             << mpView2->mPose << endl;
        setAbsPose(mpView2->mPose);
        // compute velocity
        // T_{k|k-1} = T_{k|1} * T_{1|k-1} = T_{k|1} * T_{k-1|1}^{-1}
        CamPose pose1Inv = CamPose(mpView1->mPose.getPoseInv());
        mVelocity = mpView2->mPose * pose1Inv;
        eState = State::OK;
    } else {
        mpView2->mPose = mpView1->mPose;
        setAbsPose(mpView1->mPose);
        mVelocity = CamPose();
        eState = State::LOST;
    }
    
    float meanT2VRatio = 0.0f;
    for (const shared_ptr<MapPoint>& pMPt : mpView2->mappoints()) {
        meanT2VRatio += pMPt->getTracked2VisibleRatio();
    }
    meanT2VRatio /= mvpMPts.size();
    cout << "Mean tracked-visible ratio = "
         << meanT2VRatio * 100 << "%" << endl;
    
    return eState;
}

int Tracker::poseEstimation(const CamPose& pose)
{
    // get all 3D points and corresponding 2D keypoints
    int nMatches = mvMatches2Dto3D.size();
    // give up pose estimation if number of matches is too low
    // currently restart map initialization scheme
    if (nMatches < TH_MIN_MATCHES_2D_TO_3D) {
        mpView2->mPose = mpView1->mPose;
        return 0;
    }
    Mat X3Ds, x2Ds;
    map<int, shared_ptr<MapPoint>> mpMPts = mpView2->getMPtsMap();
    for (const auto& pair : mpMPts) {
        const shared_ptr<MapPoint>& pMPt = pair.second;
        Mat X3Dt = pMPt->X3D().t();
        X3Ds.push_back(X3Dt);
        Mat x2D = Mat(mpView2->keypoint(pair.first).pt);
        Mat x2Dt = x2D.t();
        x2Ds.push_back(x2Dt);
    }
    assert(mpView2->getNumMPts() == nMatches);
    
    // compute pose by PnP algorithm (given initial pose guess as that from
    // previous frame)
    // RcwAA: 3x1 angle-axis rotation representation
    Mat RcwAA = pose.getRotationAngleAxis();
    Mat tcw = pose.getTranslation();
    Mat idxInliers = Mat::zeros(nMatches, 1, CV_32FC1);
    // TODO: use scale & octave to compute threshold
    float thReprojErr = Config::scaleFactor() * TH_REPROJ_ERR_FACTOR;
    // assume zero distortion coefficients for camera as keypoint coordinates
    // have already been undistorted
    bool bIsPoseEstimated = 
        solvePnPRansac(X3Ds, x2Ds, Config::K(), cv::noArray(),
                       RcwAA, tcw, true, 100, thReprojErr, 0.99, idxInliers,
                       cv::SOLVEPNP_ITERATIVE);
    if (bIsPoseEstimated) {
        Mat Rcw; // 3x3 rotation matrix
        cv::Rodrigues(RcwAA, Rcw, cv::noArray());
        
        mpView2->mPose.setPose(Rcw, tcw);
        // set outlier flag to map points
        int nRowInlier = 0;
        for (int i = 0; i < nMatches; ++i) {
            int nIdx = idxInliers.at<int>(nRowInlier, 0);
            //shared_ptr<MapPoint> pMPt = mvpMPts[mvMatches2Dto3D[i].trainIdx];
            shared_ptr<MapPoint> pMPt =
                mpView2->mappoint(mvMatches2Dto3D[i].queryIdx);
            assert(pMPt);
            if (i < nIdx) {
                pMPt->setOutlier(true);
            } else if (i == nIdx) {
                pMPt->setOutlier(false);
                nRowInlier++;
            }
        }
        
        // check reprojection error
        //float errorRT = 0.0f;
        //for (int i = 0; i < nMatches; ++i) {
        //    Mat xReproj = mpView2->coordWorld2Img(X3Ds.row(i).t());
        //    Mat x = x2Ds.row(i).t();
        //    Mat diffx = xReproj - x;
        //    errorRT += diffx.dot(diffx);
        //}
        //errorRT /= nMatches;
        //if (errorRT > TH_REPROJ_ERR_FACTOR*TH_REPROJ_ERR_FACTOR * 4.0f) {
        //    mpView2->mPose = mpView1->mPose;
        //    return 0;
        //}
        //cout << "Mean square reprojection error for PnP = " << errorRT << endl;

        return idxInliers.rows;
    } else {
        mpView2->mPose = mpView1->mPose;
        return 0;
    }
    return 0;
}

 std::vector<std::shared_ptr<MapPoint>> Tracker::trackLocalMap()
{
    vector<shared_ptr<MapPoint>> vpMPtsAll;
    // get all related KFs based on each matched map point
    set<shared_ptr<KeyFrame>> spKFsAll;
    vector<shared_ptr<MapPoint>> vpMPtsCur = mpView2->mappoints();
    for (const auto& pMPt : vpMPtsCur) {
        vector<shared_ptr<KeyFrame>> vpKFs = pMPt->getRelatedKFs();
        spKFsAll.insert(vpKFs.cbegin(), vpKFs.cend());
    }
    // get additional keyframes based on the latest KF
    vector<shared_ptr<KeyFrame>> vpConnectedKFs = mpKFLatest->getConnectedKFs();
    spKFsAll.insert(vpConnectedKFs.cbegin(), vpConnectedKFs.cend());
    // get all related map points
    set<shared_ptr<MapPoint>> spMPtsAll;
    for (const auto& pKF : spKFsAll) {
        vector<shared_ptr<MapPoint>> vpMPtsKF = pKF->mappoints();
        spMPtsAll.insert(vpMPtsKF.cbegin(), vpMPtsKF.cend());
    }
    vpMPtsAll.reserve(spMPtsAll.size());
    // add visible map points
    for (const auto& pMPt : spMPtsAll) {
        if (!pMPt) {
            continue;
        }
        // depth must be positive
        Mat Xc = mpView2->coordWorld2Cam(pMPt->X3D());
        float Zc = Xc.at<float>(2);
        if (Zc <= 0) {
            continue;
        }
        // reprojected point must be inside image border
        Mat xImg = mpView2->coordWorld2Img(pMPt->X3D());
        if (!Utility::is2DPtInBorder(xImg)) {
            continue;
        }
        // update visibility data
        pMPt->addCntVisible(1);
        pMPt->setIdxLastVisibleFrm(mpView2->index());
        vpMPtsAll.push_back(pMPt);
    }
    return vpMPtsAll;
}

void Tracker::updateMPtTrackedData() const
{
    vector<shared_ptr<MapPoint>> vpMPts = mpView2->mappoints();
    for (const auto& pMPt : vpMPts) {
        if (!pMPt->isOutlier()) {
            pMPt->addCntTracked(1);
        }
    }
}

bool Tracker::qualifiedAsKeyFrame(int nInliers) const
{
    bool bQualified = false;
    //const CamPose& poseCur = mpView2->mPose;
    //const CamPose& poseKFLatest = mpKFLatest->mPose;
    //// T_{n|m} = T_{n|1} T_{1|m} = T_{n|1} T_{m|1}^{-1}
    //CamPose poseKFLatestInv = poseKFLatest.getCamPoseInv();
    //CamPose poseRelative = poseCur * poseKFLatestInv;
    // check large motion between current frame and last keyframe
    if (nInliers < 150) {
        bQualified = true;
    }
    return bQualified;
}

void Tracker::addNewKeyFrame()
{
    shared_ptr<KeyFrame> pKF2 = make_shared<KeyFrame>(mpView2);
    // add observation data to each matched map point
    map<int, shared_ptr<MapPoint>> mpMPts = pKF2->getMPtsMap();
    for (auto& pair : mpMPts) {
        shared_ptr<MapPoint>& pMPt = pair.second;
        pMPt->addObservation(pKF2, pair.first);
        // update distinctive descriptor
        pMPt->updateDescriptor();
    }
    pKF2->updateConnections();
    mpLocalMapper->insertKeyFrame(pKF2);
    mpKFLatest = pKF2;
}

} // namespace SLAM_demo
