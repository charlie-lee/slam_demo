/**
 * @file   Frame.hpp
 * @brief  Header of Frame class for storing intra- and inter-frame info.
 * @author Charlie Li
 * @date   2019.08.12
 */

#include "Frame.hpp"

#include <memory>

#include <nanoflann.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp> // cv::cv2eigen()
#include <opencv2/calib3d.hpp> // cv::undistortPoints()
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <Eigen/Dense>
#include "Config.hpp"
#include "FrameBase.hpp"
#include "MapPoint.hpp"

namespace SLAM_demo {

using std::make_shared;
using std::shared_ptr;
using std::vector;
using cv::Mat;

const int Frame::NUM_BLK_X = 4;
const int Frame::NUM_BLK_Y = 4;
const int Frame::TH_EDGE = 17;
unsigned Frame::nNextIdx = 0;

Frame::Frame(const cv::Mat& img, double timestamp) :
    FrameBase(timestamp), mnIdx(nNextIdx++)
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
    // extract image features and get keypoint positions (mx2Ds)
    extractFeatures(img);
    // construct K-D tree
    mpKDTree = make_shared<nanoflannKDTree>(2 /* dim */,
                                            std::cref(*mpx2Ds),
                                            20 /* max leaf */);
    mpKDTree->index->buildIndex();
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
    // undistort keypoint coordinates and construct mx2Ds
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
    kpts = kpts.reshape(1); // reshape to 1 channel (currently 2 channels)
    // update keypoint coordinates (no out-of-border keypoint filtering)
    for (unsigned i = 0; i < mvKpts.size(); ++i) {
        mvKpts[i].pt.x = kpts.at<float>(i, 0);
        mvKpts[i].pt.y = kpts.at<float>(i, 1);
    }
    // put keypoint coordinates into mx2Ds for keypoint searching scheme
    mpx2Ds = make_shared<Eigen::MatrixX2f>();
    mpx2Ds->resize(kpts.rows, kpts.cols);
    cv::cv2eigen(kpts, *mpx2Ds);
}

} // namespace SLAM_demo
