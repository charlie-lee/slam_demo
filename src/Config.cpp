/**
 * @file   Config.hpp
 * @brief  Implementations of SLAM system configuration class.
 * @author Charlie Li
 * @date   2019.08.08
 */

#include "Config.hpp"

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include "System.hpp"

using namespace std;

namespace SLAM_demo {

bool Config::setParameters(const string& strCfgFile, System::Mode eMode)
{
    // open cfg file
    cv::FileStorage fs(strCfgFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Configuration file loading failed!" << endl;
        return false;
    }
    // set camera parameters
    setCamParams(fs, eMode);
    return true;
}

ostream& operator<<(ostream& os, const Config& cfg)
{
    os << "SLAM system parameters:" << endl;
    os << "### Camera Parameters ###" << endl;
    const vector<CameraParameters>& vCamParams = cfg.mvCamParams;
    for (int i = 0; i < vCamParams.size(); ++i) {
        os << "- View " << i << ":" << endl
           << "  - Image resolution: "
           << vCamParams[i].w << "x" << vCamParams[i].h << endl
           << "  - fx: " << vCamParams[i].fx << endl
           << "  - fy: " << vCamParams[i].fy << endl
           << "  - cx: " << vCamParams[i].cx << endl
           << "  - cy: " << vCamParams[i].cy << endl
           << "  - k1: " << vCamParams[i].k1 << endl
           << "  - k2: " << vCamParams[i].k2 << endl
           << "  - p1: " << vCamParams[i].p1 << endl
           << "  - p2: " << vCamParams[i].p2 << endl
           << "  - FPS: " << vCamParams[i].fps << endl;
    }
    return os;
}

void Config::setCamParams(const cv::FileStorage& fs, System::Mode eMode)
{
    // confirm mode and allocate camera parameters
    if (eMode == System::Mode::MONOCULAR || eMode == System::Mode::RGBD) {
        mvCamParams.resize(1);
        mvK.resize(1);
        // set camera parameters
        CameraParameters camParams;
        fs["Camera.width"] >> camParams.w;
        fs["Camera.height"] >> camParams.h;
        fs["Camera.fx"] >> camParams.fx;
        fs["Camera.fy"] >> camParams.fy;
        fs["Camera.cx"] >> camParams.cx;
        fs["Camera.cy"] >> camParams.cy;
        fs["Camera.k1"] >> camParams.k1;
        fs["Camera.k2"] >> camParams.k2;
        fs["Camera.p1"] >> camParams.p1;
        fs["Camera.p2"] >> camParams.p2;
        fs["Camera.fps"] >> camParams.fps;
        mvCamParams[0] = camParams;
        setCamIntrinsics(0);
    } else if (eMode == System::Mode::STEREO) {
        mvCamParams.reserve(2);
        // TODO: process multi-view parameters
    }
}

void Config::setCamIntrinsics(int view)
{
    cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
    K.at<double>(0, 0) = Config::fx(view);
    K.at<double>(1, 1) = Config::fy(view);
    K.at<double>(0, 2) = Config::cx(view);
    K.at<double>(1, 2) = Config::cy(view);
    K.copyTo(mvK[view]);
}

} // namespace SLAM_demo
