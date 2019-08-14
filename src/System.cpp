/**
 * @file   System.cpp
 * @brief  Implementations of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include "System.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include "Tracker.hpp"

namespace SLAM_demo {

using std::cout;
using std::endl;
using std::make_shared;
using std::string;
using std::vector;

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
    // tracker module
    mpTracker = make_shared<Tracker>(Tracker(meMode));
}

void System::trackImgs(const vector<cv::Mat>& vImgs, double timestamp)
{
    if (meMode == System::Mode::MONOCULAR) {
        mpTracker->trackImgsMono(vImgs[0], timestamp);
    } else {
        assert(0); // TODO: track images in other SLAM modes
    }
}

} // namespace SLAM_demo
