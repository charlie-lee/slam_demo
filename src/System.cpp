/**
 * @file   System.cpp
 * @brief  Implementations of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include "System.hpp"

#include <iostream>

#include <opencv2/core.hpp>

using namespace std;

namespace SLAM_demo {

System::System(Mode eMode):
    meMode(eMode)
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
}

} // namespace SLAM_demo
