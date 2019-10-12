/**
 * @file   Tracker.cpp
 * @brief  Implementations of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#include "Tracker.hpp"

#include <cmath>
#include <memory>
#include <set>
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
#include "Optimizer.hpp"

#include <iostream> // temp for debugging

namespace SLAM_demo {

using std::shared_ptr;
using std::make_shared;
using std::vector;
using cv::Mat;

using std::cout;
using std::endl;

// constants
const bool Tracker::USE_LOWE_RATIO_TEST = false;
const float Tracker::TH_RATIO_DIST = 0.7f;
const float Tracker::TH_MAX_DIST_MATCH = 64.0f;
const float Tracker::TH_MAX_RATIO_FH = 0.2f;
const float Tracker::TH_COS_PARALLAX = 0.9999f;
const float Tracker::TH_REPROJ_ERR_FACTOR = 3.0f;
const float Tracker::TH_POSE_SEL = 0.8f;
const float Tracker::TH_MIN_RATIO_TRIANG_PTS = 0.5f;
const int Tracker::TH_MIN_MATCHES_3D_TO_2D = 20;
// other data
unsigned Tracker::n1stFrame = 0;

Tracker::Tracker(System::Mode eMode, const std::shared_ptr<Map>& pMap) :
    meMode(eMode), meState(State::NOT_INITIALIZED), mbFirstFrame(true),
    mpMap(pMap), mpView1(nullptr), mpView2(nullptr)
{
    // initialize feature matcher
    if (USE_LOWE_RATIO_TEST) {
        // FLANN-based matcher & use Lowe's ratio test
        mpFeatMatcher = make_shared<cv::FlannBasedMatcher>(
            cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    } else {
        // use symmetric test
        mpFeatMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    }
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

    cout << "Extracted "
         << pFrameCur->getKeyPoints().size() << " keypoint(s)" << endl;
    cout << mpMap->getAllFrames().size() << " frames in map" << endl;
    
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
    cv::waitKey(1);
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
    Tracker::n1stFrame = mpView1->getFrameIdx();
    if (meMode == System::Mode::MONOCULAR) {
        return initializeMapMono();
    }
    return State::NOT_INITIALIZED;
}

Tracker::State Tracker::initializeMapMono()
{
    // match features between previous (1) and current (2) frame
    matchFeatures2Dto2D();

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
        shared_ptr<Optimizer> pOpt = make_shared<Optimizer>(mpMap);
        pOpt->globalBundleAdjustment(0, 10, true);
        //pOpt->frameBundleAdjustment(2, 10, true);

        // assign velocity
        mVelocity = mpView2->mPose;
 
        return State::OK;
    }
    return State::NOT_INITIALIZED; 
}

void Tracker::matchFeatures2Dto2D()
{
    mvMatches2Dto2D.clear();
    unsigned nBestMatches = USE_LOWE_RATIO_TEST ? 2 : 1;
    vector<vector<cv::DMatch>> vMatches;
    mpFeatMatcher->knnMatch(mpView1->getFeatDescriptors(), // query
                            mpView2->getFeatDescriptors(), // train
                            vMatches,
                            nBestMatches); // get 2 best matches
    // find good matches using Lowe's ratio test
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    mvMatches2Dto2D.reserve(vMatches.size());
    for (unsigned i = 0; i < vMatches.size(); ++i) {
        if (vMatches[i].size() != nBestMatches) {
            continue;
        }
        if (nBestMatches == 2 &&
            vMatches[i][0].distance >=
            TH_RATIO_DIST * vMatches[i][1].distance) {
            continue;
        }
        // filter out-of-border matches
        const cv::Point2f& pt1 = vKpts1[vMatches[i][0].queryIdx].pt;
        const cv::Point2f& pt2 = vKpts2[vMatches[i][0].trainIdx].pt;
        if (!is2DPtInBorder(Mat(pt1)) && !is2DPtInBorder(Mat(pt2))) {
            continue;
        }
        // filter matches whose dist between reprojected 2D point in view 1
        // and 2D point in view 2 is larger than a threshold
        Mat x1 = Mat(pt1);
        Mat x2 = Mat(pt2);
        Mat xDiff = x1 - x2;
        float xDistSq = xDiff.dot(xDiff);
        if (xDistSq > TH_MAX_DIST_MATCH * TH_MAX_DIST_MATCH) {
            continue;
        }
        mvMatches2Dto2D.push_back(vMatches[i][0]);            
    }
}

void Tracker::matchFeatures3Dto2D()
{
    // reset frame observation data and match data
    resetMPtObsData();
    mvMatches3Dto2D.clear();

    // get all map points
    mvpMPts = mpMap->getAllMPts();
    // prepare feature descriptors
    unsigned nBestMatches = USE_LOWE_RATIO_TEST ? 2 : 1;
    // get descriptors from the map and view 2.
    int nMPts = mvpMPts.size();
    Mat descsView2 = mpView2->getFeatDescriptors();
    Mat descsMap(nMPts, descsView2.cols, descsView2.type());
    for (int i = 0; i < nMPts; ++i) {
        (mvpMPts[i]->getDesc()).copyTo(descsMap.row(i));
    }
    // feature matching
    // query: map points; train: keypoints in current frame (mpView2)
    vector<vector<cv::DMatch>> vMatches;
    mpFeatMatcher->knnMatch(descsMap, // query set
                            descsView2, // train set
                            vMatches,
                            nBestMatches); // get 2 best matches
    //mpFeatMatcher->radiusMatch(descsMap, // query set
    //                           descsView2, // train set
    //                           vMatches,
    //                           32.f); // Hamming distance threshold
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    mvMatches3Dto2D.reserve(vMatches.size());
    for (unsigned i = 0; i < vMatches.size(); ++i) {
        if (vMatches[i].size() != nBestMatches) {
            continue;
        }
        if (nBestMatches == 2 &&
            vMatches[i][0].distance >=
            TH_RATIO_DIST * vMatches[i][1].distance) {
            continue;
        }
        // filter out-of-border matches
        const cv::Point2f& pt2 = vKpts2[vMatches[i][0].trainIdx].pt;
        if (!is2DPtInBorder(Mat(pt2))) { // only check current frame
            continue;
        }
        // filter matches whose dist between reprojected 2D point in view 1
        // and 2D point in view 2 is larger than a threshold
        shared_ptr<MapPoint> pMPt = mvpMPts[vMatches[i][0].queryIdx];
        Mat x1Reproj = mpView1->coordWorld2Img(pMPt->getX3D());
        Mat x2 = Mat(pt2);
        Mat xDiff = x1Reproj - x2;
        float xDistSq = xDiff.dot(xDiff);
        if (xDistSq > TH_MAX_DIST_MATCH * TH_MAX_DIST_MATCH) {
            continue;
        }
        mvMatches3Dto2D.push_back(vMatches[i][0]);
    }
    // update map point observation data based on current 3D-to-2D match result
    updateMPtObsData();
}

inline bool Tracker::is2DPtInBorder(const cv::Mat& pt) const
{
    bool result =
        (pt.at<float>(0) >= 0 && pt.at<float>(0) < Config::width()) &&
        (pt.at<float>(1) >= 0 && pt.at<float>(1) < Config::height());
    return result;
}

void Tracker::displayFeatMatchResult(int viewPrev, int viewCur) const
{
    // get keypoint data
    vector<cv::KeyPoint> vKpts1;
    vector<cv::KeyPoint> vKpts2 = mvpFramesCur[viewCur]->getKeyPoints();
    const vector<cv::DMatch>* pvMatches = nullptr;
    if (meState == State::NOT_INITIALIZED) {
        vKpts1 = mvpFramesPrev[viewPrev]->getKeyPoints();
        pvMatches = &mvMatches2Dto2D;
    } else if (meState == State::OK || meState == State::LOST) {
        // Construct a vector of fake keypoints for the map points
        vKpts1.resize(mvpMPts.size());
        int nMatches = mvMatches3Dto2D.size();
        // get reprojected keypoint positions based on view 1 (viewPrev)
        for (int i = 0; i < nMatches; ++i) {
            int nIdxMPt = mvMatches3Dto2D[i].queryIdx;
            shared_ptr<MapPoint> pMPt = mvpMPts[nIdxMPt];
            // reprojected image coord in view 1
            if (pMPt) {
                Mat x1 = mvpFramesPrev[viewPrev]->coordWorld2Img(
                    pMPt->getX3D());
                vKpts1[nIdxMPt] = cv::KeyPoint(
                    x1.at<float>(0), x1.at<float>(1), 1 /* size */);
            } else {
                vKpts1[nIdxMPt] = cv::KeyPoint(-1, -1, 0);
            }
        }
        pvMatches = &mvMatches3Dto2D;
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
    //    if (is2DPtInBorder(Mat(kpt.pt))) {
    //        cv::drawMarker(imgOut, kpt.pt,
    //                       cv::Scalar(127, 127, 0), // marker color
    //                       cv::MARKER_DIAMOND, // marker shape
    //                       5); // marker size
    //    }
    //}
    //for (const auto& kpt : vKpts2) {
    //    if (is2DPtInBorder(Mat(kpt.pt))) {
    //        cv::drawMarker(imgOut, kpt.pt, cv::Scalar(255, 0, 0),
    //                       cv::MARKER_SQUARE, 5);
    //    }
    //}
    // draw arrowed lines from keypoints in view 1 to view 2 on matches
    for (unsigned i = 0; i < pvMatches->size(); ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[(*pvMatches)[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[(*pvMatches)[i].trainIdx];
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
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    unsigned nMatches = mvMatches2Dto2D.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPts1, vPts2; 
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].trainIdx];
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
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    unsigned nMatches = mvMatches2Dto2D.size();
    // 2D keypoints as input of F computation function
    vector<cv::Point2f> vPts1, vPts2; 
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].trainIdx];
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
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    // get the vector of 2D keypoints from both frames
    unsigned nMatches = mvMatches2Dto2D.size();
    vector<cv::Point2f> vPts1;
    vector<cv::Point2f> vPts2;
    vPts1.reserve(nMatches);
    vPts2.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].trainIdx];
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
            const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[j].queryIdx];
            const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[j].trainIdx];
            if (checkTriangulatedPt(Xw, kpt1, kpt2, tmpPose)) {
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
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    // compute reprojection errors
    float errorF = 0.f;
    float errorH = 0.f;
    unsigned nMatches = mvMatches2Dto2D.size();
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].trainIdx];
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
                                ReprojErrScheme eScheme) const
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

bool Tracker::checkTriangulatedPt(const cv::Mat& Xw,
                                  const cv::KeyPoint& kpt1,
                                  const cv::KeyPoint& kpt2,
                                  const CamPose& pose2) const
{
    const CamPose& pose1 = mpView1->mPose;
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
    Mat O1 = pose1.getCamOrigin(); // 3D cam origin in previous frame
    Mat O2 = pose2.getCamOrigin(); // 3D cam origin in current frame
    Mat Xc1o1 = O1 - Xc1; // vector from Xc1 to o1
    Mat Xc2o2 = O2 - Xc2; // vector from Xc2 to o2
    float normXc1o1 = cv::norm(Xc1o1, cv::NORM_L2);
    float normXc2o2 = cv::norm(Xc2o2, cv::NORM_L2);
    float cosParallax = Xc1o1.dot(Xc2o2) / (normXc1o1 * normXc2o2);
    if (cosParallax > TH_COS_PARALLAX) {
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

void Tracker::buildInitialMap(const cv::Mat& Xws,
                              const std::vector<int>& vnIdxPts) const
{
    // clear the map to build a new one
    mpMap->clear();
    // traverse all triangulated points
    Mat descs2 = mpView2->getFeatDescriptors();
    for (unsigned i = 0; i < vnIdxPts.size(); ++i) {
        int nIdxKpt1 = mvMatches2Dto2D[vnIdxPts[i]].queryIdx;
        int nIdxKpt2 = mvMatches2Dto2D[vnIdxPts[i]].trainIdx;
        // Xws have all triangulated points, here only select the valid ones
        Mat Xw = Xws.col(vnIdxPts[i]); // i-th column for i-th point
        Mat desc2 = descs2.row(nIdxKpt2); // i-th row for i-th descriptor
        shared_ptr<MapPoint> pMPt = make_shared<MapPoint>(mpMap, Xw);
        // add observations for map point & frame
        pMPt->addObservation(mpView1, nIdxKpt1);
        pMPt->addObservation(mpView2, nIdxKpt2);
        mpView1->addObservation(pMPt);
        mpView2->addObservation(pMPt);
        // update distinctive descriptor
        pMPt->updateDescriptor();
        // add point into the map & update data of related frames
        mpMap->addMPt(pMPt);
    }
}

Tracker::State Tracker::track()
{
    // input:
    // - mpView1 & mpView2 with keypoints & descriptors
    // - the map with map points and their descriptors
    State eState = State::OK; 
    shared_ptr<Optimizer> pOpt = make_shared<Optimizer>(mpMap); // optimizer
    int nInliers; // map point inliers
    
    // get all map points
    mvpMPts = mpMap->getAllMPts();
    cout << "Number of map points = " << mvpMPts.size() << endl;
    
    // feature matching between the map and view 2
    matchFeatures3Dto2D();

    // pose estimation (1st)
    nInliers = poseEstimation(mVelocity * mpView1->mPose);
    cout << "Inliers/total 3D-to-2D matches (RANSAC PnP): "
         << nInliers << "/" << mvMatches3Dto2D.size() << endl;

    // only optimize pose
    nInliers = pOpt->poseOptimization(mpView2);
    cout << "Inliers/Total matches (pose BA): " << nInliers << "/"
         << mvMatches3Dto2D.size() << endl;

    //pOpt->frameBundleAdjustment(1, 5, true);
    
    // feature matching between view 1 & 2 for more correspondences
    matchFeatures2Dto2D();
    // triangulate new 3D points & fuse them into the map
    Mat Xws; // 3xN matrix containing N 3D points
    vector<int> vnIdxPts; // index of valid 3D points
    triangulate3DPts(Xws, vnIdxPts);
    fuseMPts(Xws, vnIdxPts);
    
    // feature matching between the map and view 2
    matchFeatures3Dto2D();

    // pose estimation (2nd)
    //nInliers = poseEstimation(mpView2->mPose);
    //cout << "Inliers/total 3D-to-2D matches (RANSAC PnP, 2nd): "
    //     << nInliers << "/" << mvMatches3Dto2D.size() << endl;
    
    // only optimize pose
    //nInliers = pOpt->poseOptimization(mpView2);
    //cout << "Inliers/Total matches (pose BA, 2nd): " << nInliers << "/"
    //     << mvMatches3Dto2D.size() << endl;

    nInliers = pOpt->frameBundleAdjustment(20, 10, true);
    //pOpt->globalBundleAdjustment(10, 5, true);
    cout << "Inliers/Total matches (frame BA): " << nInliers << "/"
         << mvMatches3Dto2D.size() << endl;

    // global BA every N frames
    //if ((System::nCurrentFrame - n1stFrame + 1) % 20 == 0) {
    //    pOpt->globalBundleAdjustment(20, 10, true);
    //}
    //} else {
    //    pOpt->frameBundleAdjustment(5, 10, true);
    //}

    // update visibility counter of all existed map points
    updateMPtVisibleData();

    // do not update map point data to the map if there's no
    // new triangulated map points in this frame
    if (vnIdxPts.size() < 1) {
        resetMPtVisibleData();
        resetMPtObsData();
    }
    
    // remove redundant map points
    mpMap->removeMPts();
    
    // temp test on display of feature matching result
    displayFeatMatchResult(0, 0);

    cout << mvMatches3Dto2D.size() << " matches after triangulation" << endl;

    if (mvMatches3Dto2D.size() > 10) {
        cout << "Pose T_{" << mpView2->getFrameIdx() << "|"
             << Tracker::n1stFrame  << "}:" << endl
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

    float meanO2VRatio = 0.0f;
    mvpMPts = mpMap->getAllMPts();
    for (const shared_ptr<MapPoint>& pMPt : mvpMPts) {
        meanO2VRatio += pMPt->getObs2VisibleRatio();
    }
    meanO2VRatio /= mvpMPts.size();
    cout << "mean observe-to-visible ratio for map points = "
         << meanO2VRatio * 100 << "%" << endl;
    
    return eState;
}

int Tracker::poseEstimation(const CamPose& pose)
{
    // get all 3D points and corresponding 2D keypoints
    int nMatches = mvMatches3Dto2D.size();
    // give up pose estimation if number of matches is too low
    // currently restart map initialization scheme
    if (nMatches < TH_MIN_MATCHES_3D_TO_2D) {
        mpView2->mPose = mpView1->mPose;
        return 0;
    }
    // get input 3D and 2D points for PnP
    Mat X3Ds(nMatches, 3, CV_32FC1);
    Mat x2Ds(nMatches, 2, CV_32FC1);
    vector<cv::KeyPoint> vKpts2 = mpView2->getKeyPoints();
    for (int i = 0; i < nMatches; ++i) {
        // 3D points (copy 1x3 cv::Mat)
        shared_ptr<MapPoint> pMPt = mvpMPts[mvMatches3Dto2D[i].queryIdx];
        Mat X3Dt = pMPt->getX3D().t();
        X3Dt.copyTo(X3Ds.row(i));
        // 2D keypoints (cv::Point2f -> cv::Mat, copy 1x2 cv::Mat)
        Mat x2D = Mat(vKpts2[mvMatches3Dto2D[i].trainIdx].pt);
        Mat x2Dt = x2D.t();
        x2Dt.copyTo(x2Ds.row(i));
    }
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
            shared_ptr<MapPoint> pMPt = mvpMPts[mvMatches3Dto2D[i].queryIdx];
            if (i < nIdx) {
                pMPt->setOutlier(true);
            } else if (i == nIdx) {
                pMPt->setOutlier(false);
                nRowInlier++;
            }
        }
        
        // check reprojection error
        float errorRT = 0.0f;
        for (int i = 0; i < nMatches; ++i) {
            Mat xReproj = mpView2->coordWorld2Img(X3Ds.row(i).t());
            Mat x = x2Ds.row(i).t();
            Mat diffx = xReproj - x;
            errorRT += diffx.dot(diffx);
        }
        errorRT /= nMatches;
        if (errorRT > TH_REPROJ_ERR_FACTOR*TH_REPROJ_ERR_FACTOR * 4.0f) {
            mpView2->mPose = mpView1->mPose;
            return 0;
        }
        cout << "Mean square reprojection error for PnP = " << errorRT << endl;

        return idxInliers.rows;
    } else {
        mpView2->mPose = mpView1->mPose;
        return 0;
    }
    return 0;
}

void Tracker::updateMPtVisibleData() const
{
    int nMPts = mvpMPts.size();
    // update map point visibility data
    for (int i = 0; i < nMPts; ++i) {
        const shared_ptr<MapPoint>& pMPt = mvpMPts[i];
        // already removed from the map
        if (!pMPt) {
            continue;
        }
        // visibility data already updated
        if (pMPt->getIdxLastObsFrm() == mpView2->getFrameIdx()) {
            continue;
        }
        Mat Xc = mpView2->coordWorld2Cam(pMPt->getX3D());
        float Zc = Xc.at<float>(2);
        // map point is visible if it is in front of the camera (depth > 0)
        if (Zc <= 0) {
            continue;
        }
        Mat x = mpView2->coordCam2Img(Xc);
        // projected 2D image point should be inside image border
        if (!is2DPtInBorder(x)) {
            continue;
        }
        // regard the map point as visible if passing all the tests
        pMPt->addCntVisible(1);
        pMPt->setIdxLastObsFrm(mpView2->getFrameIdx());
    }
    // update observation data
    for (unsigned i = 0; i < mvMatches3Dto2D.size(); ++i) {
        const cv::DMatch& m3Dto2D = mvMatches3Dto2D[i];
        unsigned nIdxMPt = m3Dto2D.queryIdx;
        shared_ptr<MapPoint> pMPt = mvpMPts[nIdxMPt];
        // only update data for visible map points (discard potential
        // wrong matches)
        if (pMPt->getIdxLastObsFrm() == mpView2->getFrameIdx()) {
            pMPt->updateDescriptor();
        }
    }
}

void Tracker::updateMPtObsData() const
{
    for (auto& m3Dto2D : mvMatches3Dto2D) {
        unsigned nIdxMPt = m3Dto2D.queryIdx;
        unsigned nIdxKpt = m3Dto2D.trainIdx;
        shared_ptr<MapPoint> pMPt = mvpMPts[nIdxMPt];
        mpView2->addObservation(pMPt);
        pMPt->addObservation(mpView2, nIdxKpt);
    }
}

void Tracker::resetMPtObsData() const
{
    vector<shared_ptr<MapPoint>> vpMPts = mpView2->getpMPtsObserved();
    for (auto& pMPt : vpMPts) {
        pMPt->removeObservation(mpView2);
    }
    mpView2->resetObservation();
}

void Tracker::resetMPtVisibleData() const
{
    vector<shared_ptr<MapPoint>> vpMPts = mpView2->getpMPtsObserved();
    for (auto& pMPt : vpMPts) {
        if (pMPt->getIdxLastObsFrm() == mpView2->getFrameIdx()) {
            pMPt->addCntVisible(-1);
        }
    }
}

void Tracker::triangulate3DPts(cv::Mat& Xws, std::vector<int>& vnIdxPts) const
{
    // get all 2D-to-2D matches
    const vector<cv::KeyPoint>& vKpts1 = mpView1->getKeyPoints();
    const vector<cv::KeyPoint>& vKpts2 = mpView2->getKeyPoints();
    unsigned nMatches = mvMatches2Dto2D.size();
    vector<cv::Point2f> vPts1(nMatches);
    vector<cv::Point2f> vPts2(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].trainIdx];
        vPts1[i] = kpt1.pt;
        vPts2[i] = kpt2.pt;
    }
    // get transformation matrix K[R|t] for both views
    Mat Tcw1 = Config::K() * mpView1->mPose.getPose();
    Mat Tcw2 = Config::K() * mpView2->mPose.getPose();
    // compute triangulated 3D world points (3xN)
    Mat Xw4Ds(4, nMatches, CV_32FC1);
    cv::triangulatePoints(Tcw1, Tcw2, vPts1, vPts2, Xw4Ds);
    Xw4Ds.convertTo(Xw4Ds, CV_32FC1);
    Xws = Xw4Ds.rowRange(0, 3);
    for (unsigned i = 0; i < nMatches; ++i) {
        Xws.col(i) /= Xw4Ds.at<float>(3, i);
        const cv::KeyPoint& kpt1 = vKpts1[mvMatches2Dto2D[i].queryIdx];
        const cv::KeyPoint& kpt2 = vKpts2[mvMatches2Dto2D[i].trainIdx];
        if (checkTriangulatedPt(Xws.col(i), kpt1, kpt2, mpView2->mPose)) {
            vnIdxPts.push_back(i);
        }
    }
    cout << "New triangulated good points: " << vnIdxPts.size() << "/"
         << nMatches << endl;
}

void Tracker::fuseMPts(const cv::Mat& Xws,
                       const std::vector<int>& vnIdxPts) const
{
    Mat descs2 = mpView2->getFeatDescriptors();
    // set of pointers to 3D-to-2D matches
    std::set<cv::DMatch*> spMatches3Dto2D;
    for (unsigned i = 0; i < mvMatches3Dto2D.size(); ++i) {
        cv::DMatch* pm3Dto2D = const_cast<cv::DMatch*>(&mvMatches3Dto2D[i]);
        spMatches3Dto2D.insert(pm3Dto2D);
    }
    // traverse all valid triangulated points
    for (unsigned i = 0; i < vnIdxPts.size(); ++i) {
        const cv::DMatch& m2Dto2D = mvMatches2Dto2D[vnIdxPts[i]];
        // find same landmark from 3D-to-2D matches
        bool bSameLandMark = false;
        cv::DMatch* pm3Dto2DSame = nullptr;
        for (cv::DMatch* pm3Dto2D : spMatches3Dto2D) {
            // same landmark:
            // (x_{k-1}, x_k) & (X_{k-1}, x_k) -> (x_{k-2} <-> x_{k-1} <-> x_k)
            if (m2Dto2D.trainIdx == pm3Dto2D->trainIdx) {
                bSameLandMark = true;
                pm3Dto2DSame = pm3Dto2D;
                break;
            } else { // add new map points
                bSameLandMark = false;
            }
        }
        // same landmark: fuse new data into existing map point
        // TODO: improve fusion procedure
        if (bSameLandMark) {
            shared_ptr<MapPoint> pMPt = mvpMPts[pm3Dto2DSame->queryIdx];
            // descriptor in view 2 is better matched (smaller distance)
            // than that of the corresponding map point
            // (i.e. the descriptor from view 1)
            if(m2Dto2D < *pm3Dto2DSame) {
                // update map point info using data in view 2
                //pMPt->setX3D(Xws.col(vnIdxPts[i]));
                //pMPt->setDesc(descs2.row(m2Dto2D.trainIdx));
            }
            spMatches3Dto2D.erase(pm3Dto2DSame);
        } else { // new landmark: add 3D point data to the map
            int nIdxKpt1 = m2Dto2D.queryIdx;
            int nIdxKpt2 = m2Dto2D.trainIdx;
            Mat Xw = Xws.col(vnIdxPts[i]);
            Mat desc2 = descs2.row(nIdxKpt2);
            shared_ptr<MapPoint> pMPt = make_shared<MapPoint>(mpMap, Xw);
            // add observations for map point & frame
            pMPt->addObservation(mpView1, nIdxKpt1);
            //pMPt->addObservation(mpView2, nIdxKpt2);
            mpView1->addObservation(pMPt);
            //mpView2->addObservation(pMPt);
            // update distinctive descriptor
            pMPt->updateDescriptor();
            // add point into the map & update data of related frames
            mpMap->addMPt(pMPt);
        }
    }
}

cv::Mat Tracker::computeFundamental(const CamPose& CP1,
                                    const CamPose& CP2) const
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

} // namespace SLAM_demo
