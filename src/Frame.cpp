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

namespace SLAM_demo {

using cv::Mat;
using cv::ORB;

Frame::Frame(const Mat& img, double timestamp) : mTimestamp(timestamp)
{
    // configure feature extractor
    mpFeatExtractor = ORB::create(Config::nFeatures(), Config::scaleFactor(),
                                  Config::nLevels());
    extractFeatures(img);
}

void Frame::extractFeatures(const Mat& img)
{
    mpFeatExtractor->detectAndCompute(img, cv::noArray(), mvKpts, mDescs);
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
