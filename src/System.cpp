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
#include "KeyFrame.hpp"
#include "LocalMapper.hpp"
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
unsigned System::nLostFrames = 0;
const unsigned System::TH_MAX_LOST_FRAMES = 20;

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
    mpMap = make_shared<Map>();
    // local mapper module
    mpLocalMapper = make_shared<LocalMapper>(mpMap);
    // tracker module
    mpTracker = make_shared<Tracker>(meMode, mpMap, mpLocalMapper);
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
    
    // process new keyframes & triangulate new map points.
    mpLocalMapper->run();

    Tracker::State eState = mpTracker->getState();
    if (eState == Tracker::State::NOT_INITIALIZED) {
        reset(); // reset system if it is not initialized
        saveTrajectoryRT(timestamp, CamPose());
    } else if (eState == Tracker::State::OK) {
        // save trajectory (real time)
        int nIdx = mpTracker->getIdxLastPose();
        // already initialized
        CamPose pose = mpTracker->getAbsPose(nIdx);
        saveTrajectoryRT(timestamp, pose);
        // save trajectory (final optimized)
        bool bFuseRestData = false;
        saveTrajectoryOpt(bFuseRestData);
    } else if (eState == Tracker::State::LOST) {
        if (System::nLostFrames > System::TH_MAX_LOST_FRAMES) {
            mpTracker->setState(Tracker::State::NOT_INITIALIZED);
        }            
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
        cv::Mat t = pose.getCamOrigin();
        Eigen::Quaternion<float> qR = pose.getRInvQuatEigen();
        ofs << std::fixed << std::setprecision(8)
            << ts << " "
            << t.at<float>(0) << " "
            << t.at<float>(1) << " "
            << t.at<float>(2) << " "
            << qR.x() << " "
            << qR.y() << " "
            << qR.z() << " "
            << qR.w() << " "
            << endl;
    }
}

void System::reset()
{
    // clear map
    mpMap->clear();
    // clear saved trajectory
    mmTrajectoryRT.clear();
    mmTrajectoryOpt.clear();
    // reset lost frame counter
    System::nLostFrames = 0;
}

void System::saveTrajectoryRT(double timestamp, const CamPose& pose)
{
    mmTrajectoryRT.insert({timestamp, pose});
}

void System::saveTrajectoryOpt(bool bFuseRestData)
{
    set<shared_ptr<KeyFrame>>* pspKFs;
    vector<shared_ptr<KeyFrame>> vpKFs = mpMap->getAllKFs();
    set<shared_ptr<KeyFrame>> spKFs(vpKFs.cbegin(), vpKFs.cend());
    set<shared_ptr<KeyFrame>> spKFsOpt = mpMap->transferKFsOpt();
    if (bFuseRestData) {
        // add rest poses to the optimized trajectory
        pspKFs = &spKFs;
    } else {
        pspKFs = &spKFsOpt;
    }
    //vector<shared_ptr<KeyFrame>> vpKFs = mpMap->getAllKFs();
    for (const auto& pKF : *pspKFs) {
        double timestamp = pKF->timestamp();
        CamPose pose = pKF->mPose;
        mmTrajectoryOpt.insert({timestamp, pose});
    }
}

} // namespace SLAM_demo
