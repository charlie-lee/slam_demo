/**
 * @file   Tracker.cpp
 * @brief  Implementations of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#include "Tracker.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // cv::imshow()
#include "Frame.hpp"

using namespace std;
using namespace cv;

namespace SLAM_demo {

Tracker::Tracker(System::Mode eMode):
    meMode(eMode), mbFirstFrame(true)
{
    // initialize feature matcher
    mpFeatMatcher = DescriptorMatcher::create(
        DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);
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
    Mat imgGray;
    if (img.channels() == 3) {
        cvtColor(img, imgGray, cv::COLOR_RGB2GRAY);
    } else if (img.channels() == 4) {
        cvtColor(img, imgGray, cv::COLOR_RGBA2GRAY);
    } else {
        imgGray = img.clone();
    }
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
        vector<DMatch> vMatches;
        matchFeatures(mvpFrames[0], mvpRefFrames[0], vMatches);
        
        // temp test on display of feature matching result
        Mat imgOut;
        drawMatches(imgGray, mvpFrames[0]->getKeyPoints(),
                    mPrevImg, mvpRefFrames[0]->getKeyPoints(),
                    vMatches, imgOut);
        imshow("cam0: Matches between current and previous frame", imgOut);
        waitKey();
    }
    mPrevImg = imgGray;
}

void Tracker::matchFeatures(shared_ptr<Frame> pFrame1,
                            shared_ptr<Frame> pFrame2,
                            vector<DMatch>& vMatches)
{
    mpFeatMatcher->match(pFrame1->getFeatDescriptor(),
                         pFrame2->getFeatDescriptor(),
                         vMatches);
}

} // namespace SLAM_demo
