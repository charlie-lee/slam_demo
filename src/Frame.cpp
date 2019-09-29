/**
 * @file   Frame.hpp
 * @brief  Header of Frame class for storing intra- and inter-frame info.
 * @author Charlie Li
 * @date   2019.08.12
 */

#include "Frame.hpp"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp> // cv::undistortPoints()
#include "Config.hpp"
#include "MapPoint.hpp"

namespace SLAM_demo {

using std::vector;
using cv::Mat;

unsigned Frame::nNextIdx = 0;

const int Frame::NUM_BLK_X = 4;
const int Frame::NUM_BLK_Y = 4;
const int Frame::TH_EDGE = 17;

Frame::Frame(const cv::Mat& img, double timestamp) :
    mTimestamp(timestamp), mnIdx(nNextIdx++)
{
    int numFeatScale = std::min(Config::nFeatures(),
                                std::max(1, NUM_BLK_X * NUM_BLK_Y));
    // configure feature extractor
    mpFeatExtractor = cv::ORB::create(
        Config::nFeatures() / numFeatScale, // [2, nFeatures]
        Config::scaleFactor(), Config::nLevels(),
        TH_EDGE, // edgeThreshold
        0, 2, cv::ORB::HARRIS_SCORE,
        31, // patchSize
        20); // FAST threshold
    extractFeatures(img);
}

cv::Mat Frame::coordWorld2Img(const cv::Mat& Xw) const
{
    Mat Xc = coordWorld2Cam(Xw);
    Mat x = coordCam2Img(Xc);
    return x;
}

cv::Mat Frame::coordWorld2Cam(const cv::Mat& Xw) const
{
    Mat Xc(3, 1, CV_32FC1);
    Mat Rcw = mPose.getRotation();
    Mat tcw = mPose.getTranslation();
    Xc = Rcw*Xw + tcw;
    return Xc;
}

cv::Mat Frame::coordCam2Img(const cv::Mat& Xc) const
{
    Mat x(2, 1, CV_32FC1);
    Mat K = Config::K();
    float invZc = 1.0f / Xc.at<float>(2);
    Mat x3D = invZc * K * Xc;
    x3D.rowRange(0, 2).copyTo(x);
    return x;
}

void Frame::addObservation(const std::shared_ptr<MapPoint>& pMPt)
{
    mspMPtsObs.insert(pMPt);
}

std::vector<std::shared_ptr<MapPoint>> Frame::getpMPtsObserved()
{
    vector<std::shared_ptr<MapPoint>> vMPtsObs;
    vMPtsObs.reserve(mspMPtsObs.size());
    for (const std::shared_ptr<MapPoint>& pMPt : mspMPtsObs) {
        if (pMPt) {
            vMPtsObs.push_back(pMPt);
        }
    }
    // update map point set (necessary?)
    mspMPtsObs.clear();
    mspMPtsObs = std::set<std::shared_ptr<MapPoint>>(
        vMPtsObs.begin(), vMPtsObs.end());
    return vMPtsObs;
}

void Frame::extractFeatures(const cv::Mat& img)
{
    // block-based feature extraction
    mvKpts.reserve(Config::nFeatures());
    int nImgWidth = img.cols;
    int nImgHeight = img.rows;
    // expand input image by 2*TH_EDGE
    Mat imgFull;
    cv::copyMakeBorder(img, imgFull, TH_EDGE, TH_EDGE, TH_EDGE, TH_EDGE,
                       cv::BORDER_REFLECT_101);
    int nBlkHeight = nImgHeight / NUM_BLK_Y;
    for (int i = 0; i < NUM_BLK_Y; ++i) {
        int nBlkWidth = nImgWidth / NUM_BLK_X;
        int nBlkTLY = nBlkHeight * i; // top-left y coord of a block
        if (i == NUM_BLK_Y - 1) {
            nBlkHeight = nImgHeight - (NUM_BLK_Y - 1) * nBlkHeight;
        }
        for (int j = 0; j < NUM_BLK_X; ++j) {
            int nBlkTLX = nBlkWidth * j; // top-left x coord of a block
            if (j == NUM_BLK_X - 1) {
                nBlkWidth = nImgWidth - (NUM_BLK_X - 1) * nBlkWidth;
            }
            // get image ROI of the target block
            int nW = nBlkWidth + 2*TH_EDGE;
            int nH = nBlkHeight + 2*TH_EDGE;
            // copy border if ROI exceeds image border
            cv::Rect roi = cv::Rect(nBlkTLX, nBlkTLY, nW, nH);
            Mat imgROI = imgFull(roi).clone();
            
            // feature extraction (multi passes)
            vector<cv::KeyPoint> vKpts;
            Mat descs;
            //mpFeatExtractor->detectAndCompute(
            //    imgROI, cv::noArray(), vKpts, descs);
            // re-extract features using half FAST threhold
            // if there are not enough features
            std::shared_ptr<cv::ORB> pORB =
                std::dynamic_pointer_cast<cv::ORB>(mpFeatExtractor);
            int thFAST = pORB->getFastThreshold();
            int thFASTtmp = thFAST;
            while (vKpts.empty() && thFASTtmp > 10) {
                pORB->detectAndCompute(
                    imgROI, cv::noArray(), vKpts, descs);
                thFASTtmp -= 5;
                pORB->setFastThreshold(thFASTtmp);
            }
            pORB->setFastThreshold(thFAST); // restore FAST threhold value
            
            // map local keypoint coords to global image coords
            for (cv::KeyPoint& kpt : vKpts) {
                kpt.pt.x += nBlkTLX - TH_EDGE;
                kpt.pt.y += nBlkTLY - TH_EDGE;
            }
            // append features in the ROI to global containers
            mvKpts.insert(std::end(mvKpts), std::begin(vKpts), std::end(vKpts));
            if (i == 0 && j == 0) {
                descs.copyTo(mDescs);
            } else {
                mDescs.push_back(descs.clone());
            }
        }
    }
    // undistort keypoint coordinates
    undistortKpts();
}

void Frame::undistortKpts()
{
    // convert src kpts data to Nx2 matrix
    Mat kpts(mvKpts.size(), 2, CV_32F); // input of cv::undistortPoints() is Nx2
    for (unsigned i = 0; i < mvKpts.size(); ++i) {
        kpts.at<float>(i, 0) = mvKpts[i].pt.x;
        kpts.at<float>(i, 1) = mvKpts[i].pt.y;
    }
    // undistort keypoints
    cv::undistortPoints(kpts, kpts, Config::K(), Config::distCoeffs(),
                        cv::noArray(), Config::K());
    kpts.reshape(1); // reshape output to 1 channel (currently 2 channels)
    // update keypoint coordinates (no out-of-border keypoint filtering)
    for (unsigned i = 0; i < mvKpts.size(); ++i) {
        mvKpts[i].pt.x = kpts.at<float>(i, 0);
        mvKpts[i].pt.y = kpts.at<float>(i, 1);
    }
}

} // namespace SLAM_demo
