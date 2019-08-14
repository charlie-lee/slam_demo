/**
 * @file   Tracker.cpp
 * @brief  Implementations of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#include "Tracker.hpp"

#include <memory>
#include <vector>

#include <opencv2/calib3d.hpp> // cv::undistort()
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // cv::imshow()
#include "Config.hpp"
#include "Frame.hpp"

namespace SLAM_demo {

using std::shared_ptr;
using std::make_shared;
using std::vector;
using cv::Mat;

Tracker::Tracker(System::Mode eMode) : meMode(eMode), mbFirstFrame(true)
{
    // initialize feature matcher
    mpFeatMatcher = cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);
    // allocate space for vectors
    if (meMode == System::Mode::MONOCULAR) {
        mvpFrames.resize(1);
        mvpRefFrames.resize(1);
    } else if (meMode == System::Mode::STEREO || meMode == System::Mode::RGBD) {
        mvpFrames.resize(2);
        mvpRefFrames.resize(2);
    }
}

void Tracker::trackImgsMono(const Mat& img, double timestamp)
{
    // RGB -> Grayscale
    Mat imgGray = rgb2Gray(img);
    // initialize each frame
    shared_ptr<Frame> pFrame = make_shared<Frame>(Frame(imgGray));
    // feature matching
    if (mbFirstFrame) {
        mvpFrames[0] = pFrame;
        mbFirstFrame = false;
    } else {
        mvpRefFrames[0] = mvpFrames[0];
        mvpFrames[0] = pFrame;
        // match features between current (1) and reference (2) frame
        vector<cv::DMatch> vMatches;
        matchFeatures(mvpFrames[0], mvpRefFrames[0], vMatches);
        // temp test on display of feature matching result
        displayFeatMatchResult(imgGray, vMatches);
    }
    mImgPrev = imgGray;
}

Mat Tracker::rgb2Gray(const Mat& img)
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

void Tracker::matchFeatures(shared_ptr<Frame> pFrame1,
                            shared_ptr<Frame> pFrame2,
                            vector<cv::DMatch>& vMatches,
                            const float TH_DIST)
{
    vector<vector<cv::DMatch>> vknnMatches;
    mpFeatMatcher->knnMatch(pFrame1->getFeatDescriptors(),
                            pFrame2->getFeatDescriptors(),
                            vknnMatches, 2); // get 2 best matches
    // find good matches using Lowe's ratio test
    const float TH = TH_DIST;
    vMatches.reserve(vknnMatches.size());
    for (int i = 0; i < vknnMatches.size(); ++i) {
        if (vknnMatches[i][0].distance < TH * vknnMatches[i][1].distance) {
            vMatches.push_back(vknnMatches[i][0]);
        }
    }
}

void Tracker::displayFeatMatchResult(const Mat& img,
                                     const vector<cv::DMatch> vMatches)
{
    Mat imgUnD, imgPrevUnD, imgOut;
    // undistort input images
    imgUnD = img;
    imgPrevUnD = mImgPrev;
    //cv::undistort(img, imgUnD, Config::K(), Config::distCoeffs());
    //cv::undistort(mImgPrev, imgPrevUnD, Config::K(), Config::distCoeffs());
    // display keypoint matches (undistorted) on undistorted images
    cv::drawMatches(imgUnD, mvpFrames[0]->getKeyPoints(),
                    imgPrevUnD, mvpRefFrames[0]->getKeyPoints(),
                    vMatches, imgOut,
                    cv::Scalar({255, 0, 0}), // color for matching line (BGR)
                    cv::Scalar({0, 255, 0})); // color for keypoint (BGR)
    cv::imshow("cam0: Matches between current (left) and "
               "previous (right) frame", imgOut);
    cv::waitKey();
}

} // Namespace SLAM_demo
