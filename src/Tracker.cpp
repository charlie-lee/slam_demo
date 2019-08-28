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
const float Tracker::TH_SIMILARITY = 0.3f;
const float Tracker::TH_COS_PARALLAX = 0.9999f;
const float Tracker::TH_POSE_SEL = 0.99f;
const float Tracker::TH_MIN_RATIO_TRIANG_PTS = 0.5f;

Tracker::Tracker(System::Mode eMode, const std::shared_ptr<Map>& pMap) :
    meMode(eMode), mState(State::NOT_INITIALIZED), mbFirstFrame(true),
    mpMap(pMap)
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
        setAbsPose(pFrameCur->mPose);
    } else {
        mvpFramesPrev[0] = mvpFramesCur[0];
        mvpFramesCur[0] = pFrameCur;
        if (mState == State::NOT_INITIALIZED) {
            mState = initializeMap();
        } else if (mState == State::OK) {
            ; // TODO: tracking scheme after map is initialized
        } else if (mState == State::LOST) {
            ; // TODO: relocalization?
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

Tracker::State Tracker::initializeMap()
{
    if (meMode == System::Mode::MONOCULAR) {
        const shared_ptr<Frame>& pFPrev = mvpFramesPrev[0];
        const shared_ptr<Frame>& pFCur = mvpFramesCur[0];
        return initializeMapMono(pFPrev, pFCur);
    }
    return State::NOT_INITIALIZED;
}

Tracker::State Tracker::initializeMapMono(const shared_ptr<Frame>& pFPrev,
                                          const shared_ptr<Frame>& pFCur)
{
    // match features between previous (1) and current (2) frame
    vector<cv::DMatch> vMatches = matchFeatures2Dto2D(pFPrev, pFCur);
    // temp test on display of feature matching result
    displayFeatMatchResult(vMatches, 0, 0);
    // compute fundamental matrix F (from previous (p) to current (c))
    Mat Fcp = computeFundamental(pFPrev, pFCur, vMatches);
    // compute homography H (from previous (p) to current (c))
    Mat Hcp = computeHomography(pFPrev, pFCur, vMatches);
    //cout << "Fcp = " << endl << Fcp << endl << "Hcp = " << endl << Hcp << endl;
    // recover pose from F or H
    Mat Xw3Ds;
    vector<unsigned> vIdxPts;
    if (recoverPoseFromFH(pFPrev, pFCur, vMatches, Fcp, Hcp, Xw3Ds, vIdxPts)) {
        cout << "Number of triangulated points (good/total) = "
             << vIdxPts.size() << "/" << Xw3Ds.cols << endl;
        cv::waitKey();
        ; // TODO: initialize 3D map points by triangulating 2D keypoints
    }
    // temp: always return false as the initalization scheme is incomplete
    return State::NOT_INITIALIZED; 
}

vector<cv::DMatch> Tracker::matchFeatures2Dto2D(
    const shared_ptr<Frame>& pFPrev, const shared_ptr<Frame>& pFCur) const
{
    vector<cv::DMatch> vMatches;
    vector<vector<cv::DMatch>> vknnMatches;
    mpFeatMatcher->knnMatch(pFPrev->getFeatDescriptors(), // query
                            pFCur->getFeatDescriptors(), // train
                            vknnMatches, 2); // get 2 best matches
    // find good matches using Lowe's ratio test
    const vector<cv::KeyPoint>& vKptsP = pFPrev->getKeyPoints();
    const vector<cv::KeyPoint>& vKptsC = pFCur->getKeyPoints();
    vMatches.reserve(vknnMatches.size());
    for (unsigned i = 0; i < vknnMatches.size(); ++i) {
        if (vknnMatches[i][0].distance < TH_DIST * vknnMatches[i][1].distance) {
            // filter out-of-border matches
            const cv::Point2f& ptP = vKptsP[vknnMatches[i][0].queryIdx].pt;
            const cv::Point2f& ptC = vKptsC[vknnMatches[i][0].trainIdx].pt;
            if (is2DPtInBorder(Mat(ptP)) && is2DPtInBorder(Mat(ptC))) {
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

void Tracker::displayFeatMatchResult(const vector<cv::DMatch>& vMatches,
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

Mat Tracker::computeFundamental(const shared_ptr<Frame>& pFPrev,
                                const shared_ptr<Frame>& pFCur,
                                const vector<cv::DMatch>& vMatches) const
{
    // fundamental matrix result (from previous (p) to current (c))
    Mat Fcp;
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
    Fcp.convertTo(Fcp, CV_32FC1); // maintain precision
    return Fcp;
}

Mat Tracker::computeHomography(const shared_ptr<Frame>& pFPrev,
                               const shared_ptr<Frame>& pFCur,
                               const vector<cv::DMatch>& vMatches) const
{
    // fundamental matrix result (from previous (p) to current (c))
    Mat Hcp;
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
    Hcp.convertTo(Hcp, CV_32FC1);
    return Hcp;
}

bool Tracker::recoverPoseFromFH(const shared_ptr<Frame>& pFPrev,
                                const shared_ptr<Frame>& pFCur,
                                const vector<cv::DMatch>& vMatches,
                                const Mat& Fcp,
                                const Mat& Hcp,
                                cv::Mat& Xw3Ds,
                                std::vector<unsigned>& vIdxGoodPts)
{
    // select the better transformation from either F or H
    FHResult resFH = selectFH(pFPrev, pFCur, vMatches, Fcp, Hcp);
    const Mat K = Config::K(); // cam intrinsics
    vector<Mat> vRcps; // possible rotations
    vector<Mat> vtcps; // possible translations
    int nSolutions = 0; // number of possible poses
    // decompose F or H into [R|t]s
    //if (resFH == FHResult::F) {
    //    // 4 solutions
    //    nSolutions = decomposeFforRT(Fcp, vRcps, vtcps);
    //} else if (resFH == FHResult::H) {
    //    vector<Mat> vNormalcps; // plane normal matrices
    //    // at most 8 solutions
    //    nSolutions = decomposeHforRT(Hcp, vRcps, vtcps, vNormalcps);
    //} else if (resFH == FHResult::NONE) {
    //    return false;
    //}

    // temp test 20190828: traverse all [R|t] solutions
    vector<Mat> vRcpsF, vtcpsF, vRcpsH, vtcpsH, vNormalcpsH;
    int nSolF = decomposeFforRT(Fcp, vRcpsF, vtcpsF);
    int nSolH = decomposeHforRT(Hcp, vRcpsH, vtcpsH, vNormalcpsH);
    vRcps.reserve(nSolF + nSolH);
    vtcps.reserve(nSolF + nSolH);
    for (int i = 0; i < nSolF; ++i) {
        vRcps.push_back(vRcpsF[i]);
        vtcps.push_back(vtcpsF[i]);
    }
    for (int i = 0; i < nSolH; ++i) {
        vRcps.push_back(vRcpsH[i]);
        vtcps.push_back(vtcpsH[i]);
    }
    nSolutions = nSolF + nSolH;
  
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKptsP = pFPrev->getKeyPoints();
    const vector<cv::KeyPoint>& vKptsC = pFCur->getKeyPoints();
    // get the vector of 2D keypoints from both frames
    unsigned nMatches = vMatches.size();
    vector<cv::Point2f> vPtsP;
    vector<cv::Point2f> vPtsC;
    vPtsP.reserve(nMatches);
    vPtsC.reserve(nMatches);
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kptP = vKptsP[vMatches[i].queryIdx];
        const cv::KeyPoint& kptC = vKptsC[vMatches[i].trainIdx];
        vPtsP.push_back(kptP.pt);
        vPtsC.push_back(kptC.pt);
    }
    // get best pose & corresponding triangulated 3D points from the solutions
    vector<int> vnGoodPts(nSolutions, 0);
    vector<Mat> vXws(nSolutions); // triangulated 3D points
    // indices of good triangulated points for all solutions
    vector<vector<unsigned>> vvIdxPts(nSolutions);
    int nMaxGoodPts = 0;
    int nIdxBestPose = -1;
    for (int i = 0; i < nSolutions; ++i) {
        vvIdxPts[i].reserve(nMatches);
        // maintain precision
        vRcps[i].convertTo(vRcps[i], CV_32FC1);
        vtcps[i].convertTo(vtcps[i], CV_32FC1);
        // pose T = K[I|0] for previous frame
        Mat TcwP = Mat::zeros(3, 4, CV_32FC1);
        K.copyTo(TcwP.rowRange(0, 3).colRange(0, 3));
        // pose T = K[R|t] for current frame
        Mat TcwC = Mat::zeros(3, 4, CV_32FC1);
        vRcps[i].copyTo(TcwC.rowRange(0, 3).colRange(0, 3));
        vtcps[i].copyTo(TcwC.rowRange(0, 3).col(3));
        TcwC = K * TcwC;
        // compute triangulated 3D world points
        Mat Xw4Ds;
        cv::triangulatePoints(TcwP, TcwC, vPtsP, vPtsC, Xw4Ds);
        Xw4Ds.convertTo(Xw4Ds, CV_32FC1);
        Mat Xws = Xw4Ds.rowRange(0, 3);
        for (int j = 0; j < Xw4Ds.cols; ++j) {
            Xws.col(j) /= Xw4Ds.at<float>(3, j);
        }
        vXws[i] = Xws;
        // check number of good points
        CamPose tmpPose = CamPose(vRcps[i], vtcps[i]);
        for (unsigned j = 0; j < nMatches; ++j) {
            Mat Xw = Xws.col(j);
            const cv::KeyPoint& kptP = vKptsP[vMatches[j].queryIdx];
            const cv::KeyPoint& kptC = vKptsC[vMatches[j].trainIdx];
            if (checkTriangulatedPt(Xw, kptP, kptC, tmpPose)) {
                vvIdxPts[i].push_back(j);
                vnGoodPts[i]++;
            }
        }
        if (vnGoodPts[i] > nMaxGoodPts) {
            nMaxGoodPts = vnGoodPts[i];
            nIdxBestPose = i;
        }
        cout << "solution " << i << ": " << vnGoodPts[i] << endl;
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
    // TODO: store best recovered pose & triangulated 3D points
    if (nBestPose == 1) {
        // store pose & triangulated points
        pFCur->mPose.setPose(vRcps[nIdxBestPose], vtcps[nIdxBestPose]);
        vXws[nIdxBestPose].copyTo(Xw3Ds);
        vIdxGoodPts = vvIdxPts[nIdxBestPose];
        
        unsigned nCurFrmIdx = pFCur->getFrameIdx();
        cout << "Pose T_{" << nCurFrmIdx << "|" << nCurFrmIdx - 1
             <<"} recovered from " << (resFH == FHResult::F ? "F" : "H")
             << endl;// << pFCur->mPose.getPose() << endl;
        cout << pFCur->mPose << endl;
        
        return true;
    }
    return false;
}

Tracker::FHResult Tracker::selectFH(const shared_ptr<Frame>& pFPrev,
                                    const shared_ptr<Frame>& pFCur,
                                    const vector<cv::DMatch>& vMatches,
                                    const Mat& Fcp,
                                    const Mat& Hcp) const
{
    // compute reprojection errors for F & H result:
    // error_F = (1/N) * sigma_i(d(x_{2,i}, F_{21} x_{1,i})^2 +
    //                           d(x_{1,i}, F_{21}^T x_{2,i})^2
    // note: error_F is the mean distance between point and epipolar line
    // error_H = (1/N) * sigma_i(d(x_{1,i}, H_{21} x_{1,i})^2 +
    //                           d(x_{2,i}, H21^{-1} x_{2,i})^2)
    Mat Fpc, Hpc; // inverse matrix of H and F
    Fpc = Fcp.t();
    Hpc = Hcp.inv(cv::DECOMP_LU);
    // retrieve input 2D-2D matches
    const vector<cv::KeyPoint>& vKptsP = pFPrev->getKeyPoints();
    const vector<cv::KeyPoint>& vKptsC = pFCur->getKeyPoints();
    // compute reprojection errors
    float errorF = 0.f;
    float errorH = 0.f;
    unsigned nMatches = vMatches.size();
    for (unsigned i = 0; i < nMatches; ++i) {
        const cv::KeyPoint& kptP = vKptsP[vMatches[i].queryIdx];
        const cv::KeyPoint& kptC = vKptsC[vMatches[i].trainIdx];
        // use homogeneous coordinate
        Mat xP = (cv::Mat_<float>(3, 1) << kptP.pt.x, kptP.pt.y, 1.f);
        Mat xC = (cv::Mat_<float>(3, 1) << kptC.pt.x, kptC.pt.y, 1.f);
        errorF += computeReprojErr(Fcp, Fpc, xP, xC, ReprojErrScheme::F);
        errorH += computeReprojErr(Hcp, Hpc, xP, xC, ReprojErrScheme::H);
    }
    errorF /= static_cast<float>(nMatches);
    errorH /= static_cast<float>(nMatches);
    cout << "eF = " << errorF << " | eH = " << errorH << " -> ";
    // similarity between error of F and H
    float errFHSimilarity = errorF / errorH;
    if (errFHSimilarity > TH_SIMILARITY) {
        cout << "H" << endl;
        return FHResult::H;
        //} else if (errFHSimilarity < 1.f/TH_SIMILARITY) {
    } else {
        cout << "F" << endl;
        return FHResult::F;
    }
    // no result if errors are similar between F and H
    cout << "None" << endl;
    return FHResult::NONE;
}

int Tracker::decomposeFforRT(const cv::Mat& Fcp,
                             std::vector<cv::Mat>& vRcps,
                             std::vector<cv::Mat>& vtcps) const
{
    // ref: Richard Hartley and Andrew Zisserman. Multiple view geometry
    // in computer vision. Cambridge university press, 2003.
    // 4 possibilities
    vRcps.resize(4);
    vtcps.resize(4);
    // E_{cp} = K_c^T  F_{cp} K_p (K_c == K_p == K)
    Mat K = Config::K();
    Mat Ecp = K.t() * Fcp * K;
    Mat Rcp1, Rcp2, tcp;
    cv::decomposeEssentialMat(Ecp, Rcp1, Rcp2, tcp);
    // solution 1: [R_1|t]
    vRcps[0] = Rcp1.clone();
    vtcps[0] = tcp.clone();
    // solution 2: [R_1|-t]
    vRcps[1] = Rcp1.clone();
    vtcps[1] = -tcp.clone();
    // solution 3: [R_2|t]
    vRcps[2] = Rcp2.clone();
    vtcps[2] = tcp.clone();
    // solution 4: [R_2|-t]
    vRcps[3] = Rcp2.clone();
    vtcps[3] = -tcp.clone();
    return 4; // 4 possible solutions
}

int Tracker::decomposeHforRT(const cv::Mat& Hcp,
                             std::vector<cv::Mat>& vRcps,
                             std::vector<cv::Mat>& vtcps,
                             std::vector<cv::Mat>& vNormalcps) const
{
    // ref: Ezio Malis, Manuel Vargas, and others.
    // Deeper understanding of the homography decomposition for
    // vision-based control. 2007.
    // at most 8 possibilities
    vRcps.reserve(8);
    vtcps.reserve(8);
    vNormalcps.reserve(8);
    // return number of possibilities of [R|t]
    return cv::decomposeHomographyMat(Hcp, Config::K(), vRcps, vtcps,
                                      vNormalcps);
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
                                  const cv::KeyPoint& kptP,
                                  const cv::KeyPoint& kptC,
                                  const CamPose& pose) const
{
    // world coord in current frame / cam coord in previous frame
    const Mat& XcP = Xw;
    // 3D cam coord in Current frame
    Mat Rcw = pose.getRotation();
    Mat tcw = pose.getTranslation();
    Mat XcC = Rcw*XcP + tcw;
    // condition 1: must be positive depth in both views
    float invDepthP = 1.f / XcP.at<float>(2);
    float invDepthC = 1.f / XcC.at<float>(2);
    if (invDepthP < 0 || invDepthC < 0) { 
        return false;
    }
    // condition 2: the parallax of 2 views must not be too small
    Mat OP = Mat::zeros(3, 1, CV_32FC1); // 3D cam origin in previous frame
    Mat OC = pose.getCamOrigin(); // 3D cam origin in current frame
    Mat XwoP = OP - XcP; // vector from Xw to oP
    Mat XwoC = OC - XcP; // vector from Xw to oC
    float normXwoP = cv::norm(XwoP, cv::NORM_L2);
    float normXwoC = cv::norm(XwoC, cv::NORM_L2);
    float cosParallax = XwoP.dot(XwoC) / (normXwoP * normXwoC);
    if (cosParallax > TH_COS_PARALLAX) {
        return false;
    }
    // condition 3: reprojected 2D point needs to be inside image border
    const Mat K = Config::K();
    // projected cam coords in previous and current frames
    Mat XcPProj = invDepthP * K * XcP; 
    Mat XcCProj = invDepthC * K * XcC;
    // reprojected 2D image coords in both frames
    Mat xPReproj = XcPProj.rowRange(0, 2);
    Mat xCReproj = XcCProj.rowRange(0, 2);
    if (!is2DPtInBorder(xPReproj) || !is2DPtInBorder(xCReproj)) {
        return false;
    }
    // condition 4: reprojection error needs to be lower than a threshold
    //              for both views
    Mat xP = Mat(kptP.pt);
    Mat xC = Mat(kptC.pt);
    // reprojection error threshold: s^o (d < s^o)
    float s = Config::scaleFactor();
    int oP = kptP.octave;
    int oC = kptC.octave;
    float thPReproj = std::pow(s, oP);
    float thCReproj = std::pow(s, oC);
    float errPReproj = cv::norm(xP - xPReproj, cv::NORM_L2);
    float errCReproj = cv::norm(xC - xCReproj, cv::NORM_L2);
    if (errPReproj > thPReproj || errCReproj > thCReproj) {
        return false;
    }
    return true; // triangulated result is good if all conditions are met
}

} // Namespace SLAM_demo
