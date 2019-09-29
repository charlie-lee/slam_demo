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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry> // for Eigen::Quaternion
#include "CamPose.hpp"
#include "Frame.hpp"
#include "Map.hpp"
#include "Tracker.hpp"

namespace SLAM_demo {

using std::cout;
using std::endl;
using std::make_shared;
using std::map;
using std::set;
using std::shared_ptr;
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

    // save trajectory (real time)
    int nIdx = mpTracker->getIdxLastPose();
    // just initialized
    if (nIdx == 1) { //
        // reset trajectory
        mmTrajectoryRT.clear();
        mmTrajectoryOpt.clear();
        saveTrajectoryRT(timestamp, CamPose());
    }
    
    if (nIdx > 0) {
        // already initialized
        CamPose pose = mpTracker->getAbsPose(nIdx);
        saveTrajectoryRT(timestamp, pose);
        // save trajectory (final optimized)
        bool bFuseRestData = false;
        saveTrajectoryOpt(bFuseRestData);
    } else {
        // reset trajectory
        mmTrajectoryRT.clear();
        mmTrajectoryOpt.clear();
    }
}

void System::dumpTrajectory(DumpMode eMode)
{
    // add rest poses to the optimized trajectory
    std::ofstream ofs;
    const map<double, CamPose>* pTrajectory;
    if (eMode == DumpMode::REAL_TIME) {
        ofs.open("trajectoryRT.txt");
        pTrajectory = &mmTrajectoryRT;
    } else if (eMode == DumpMode::OPTIMIZED) {
        ofs.open("trajectoryOpt.txt");
        bool bFuseRestData = true;
        saveTrajectoryOpt(bFuseRestData);
        pTrajectory = &mmTrajectoryOpt;
    }
    // format: timestamp tx ty tz qx qy qz qw
    for (const auto& pair : *pTrajectory) {
        const double& ts = pair.first;
        const CamPose& pose = pair.second;
        cv::Mat tcw = pose.getTranslation();
        Eigen::Quaternion<float> qRcw = pose.getRQuatEigen();
        ofs << std::fixed << std::setprecision(8)
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

void System::saveTrajectoryRT(double timestamp, const CamPose& pose)
{
    mmTrajectoryRT.insert({timestamp, pose});
}

void System::saveTrajectoryOpt(bool bFuseRestData)
{
    set<shared_ptr<Frame>>* pspFrames;
    vector<shared_ptr<Frame>> vpRFrames = mpMap->getAllFrames();
    set<shared_ptr<Frame>> spRFrames(vpRFrames.begin(), vpRFrames.end());
    set<shared_ptr<Frame>> spFramesOpt = mpMap->transferFramesOpt();
    if (bFuseRestData) {
        // add rest poses to the optimized trajectory
        pspFrames = &spRFrames;
    } else {
        pspFrames = &spFramesOpt;
    }
    for (const auto& pFrame : *pspFrames) {
        double timestamp = pFrame->getTimestamp();
        CamPose pose = pFrame->mPose;
        mmTrajectoryOpt.insert({timestamp, pose});
    }
}

} // namespace SLAM_demo
