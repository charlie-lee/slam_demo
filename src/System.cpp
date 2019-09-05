/**
 * @file   System.cpp
 * @brief  Implementations of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include "System.hpp"

#include <iostream>
#include <iomanip> // std::setprecision()
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry> // for Eigen::Quaternion
#include "CamPose.hpp"
#include "Map.hpp"
#include "Tracker.hpp"

namespace SLAM_demo {

using std::cout;
using std::endl;
using std::make_shared;
using std::string;
using std::vector;

unsigned System::nCurrentFrame = 0;

System::System(Mode eMode) : meMode(eMode)
{
    // determine system mode
    string strMode;
    if (meMode == Mode::MONOCULAR) {
        strMode = "monocular";
    } else if (meMode == Mode::STEREO) {
        strMode = "stereo";
    } else if (meMode == Mode::RGBD) {
        strMode = "RGB-D";
    }
    cout << "SLAM demo in " << strMode << " mode." << endl;

    // initialize SLAM modules
    // map module
    mpMap = make_shared<Map>(Map());
    // tracker module
    mpTracker = make_shared<Tracker>(Tracker(meMode, mpMap));
}

void System::trackImgs(const std::vector<cv::Mat>& vImgs, double timestamp)
{
    // temp display
    cout << std::fixed << endl << "[Timestamp " << timestamp << "s]" << endl;
    
    if (meMode == Mode::MONOCULAR) {
        mpTracker->trackImgsMono(vImgs[0], timestamp);
        nCurrentFrame++;
    } else {
        assert(0); // TODO: track images in other SLAM modes
    }
    int nIdx = mpTracker->getIdxLastPose();
    CamPose pose = mpTracker->getAbsPose(nIdx);
    saveTrajectory(timestamp, pose);
}

void System::dumpTrajectory() const
{
    // format: timestamp tx ty tz qx qy qz qw
    std::ofstream ofs("trajectory.txt");
    for (const auto pair : mvTrajectory) {
        const double& ts = pair.first;
        const CamPose& pose = pair.second;
        cv::Mat tcw = pose.getTranslation();
        Eigen::Quaternion<float> qRcw = pose.getRQuatEigen();
        ofs << std::fixed << std::setprecision(4)
            << ts << " "
            << tcw.at<float>(0) << " "
            << tcw.at<float>(1) << " "
            << tcw.at<float>(2) << " "
            << qRcw.x() << " "
            << qRcw.y() << " "
            << qRcw.z() << " "
            << qRcw.w() << " "
            << endl;
    }
}

void System::saveTrajectory(double timestamp, const CamPose& pose)
{
    mvTrajectory.push_back(std::make_pair(timestamp, pose));
}

} // namespace SLAM_demo
