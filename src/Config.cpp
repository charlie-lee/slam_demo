/**
 * @file   Config.hpp
 * @brief  Implementations of SLAM system configuration class.
 * @author Charlie Li
 * @date   2019.08.08
 */

#include "Config.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include "System.hpp"

namespace SLAM_demo {

using std::cerr;
using std::endl;
using std::ostream;
using std::string;
using std::vector;

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
    // set feature extraction parameters
    setFeatExtParams(fs);
    return true;
}

void Config::setCamParams(const cv::FileStorage& fs, System::Mode eMode)
{
    // confirm mode and allocate camera parameters
    if (eMode == System::Mode::MONOCULAR || eMode == System::Mode::RGBD) {
        mvCamParams.resize(1);
        mvK.resize(1);
        mvDistCoeffs.resize(1);
        mvnDistCoeffs.resize(1);
        // set camera parameters
        CameraParameters camParams;
        fs["Camera.width"] >> camParams.w;
        fs["Camera.height"] >> camParams.h;
        fs["Camera.fps"] >> camParams.fps;
        fs["Camera.fx"] >> camParams.fx;
        fs["Camera.fy"] >> camParams.fy;
        fs["Camera.cx"] >> camParams.cx;
        fs["Camera.cy"] >> camParams.cy;
        // distortion coefficients (4, 5, 8, 12, or 14)
        mvnDistCoeffs[0] = 4; // at least 4 coefficients
        fs["Camera.k1"] >> camParams.k1;
        fs["Camera.k2"] >> camParams.k2;
        fs["Camera.p1"] >> camParams.p1;
        fs["Camera.p2"] >> camParams.p2;
        if (fs["Camera.k3"].isNamed()) {
            fs["Camera.k3"] >> camParams.k3;
            mvnDistCoeffs[0] = 5;
        }
        mvCamParams[0] = camParams;
        setCamIntrinsics(0); // set K of view 0
        setCamDistCoeffs(0); // set distortion coefficients of view 0
    } else if (eMode == System::Mode::STEREO) {
        mvCamParams.reserve(2);
        mvK.resize(1);
        mvDistCoeffs.resize(1);
        mvnDistCoeffs.resize(1);
        // TODO: process multi-view parameters
    }
}

void Config::setFeatExtParams(const cv::FileStorage& fs)
{
    fs["Feature.nFeatures"] >> mFeatParams.nFeatures;
    fs["Feature.scaleFactor"] >> mFeatParams.scaleFactor;
    fs["Feature.nLevels"] >> mFeatParams.nLevels;
}

void Config::setCamIntrinsics(int view)
{
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = Config::fx(view);
    K.at<float>(1, 1) = Config::fy(view);
    K.at<float>(0, 2) = Config::cx(view);
    K.at<float>(1, 2) = Config::cy(view);
    K.copyTo(mvK[view]);
}

void Config::setCamDistCoeffs(int view)
{
    cv::Mat distCoeffs(mvnDistCoeffs[view], 1, CV_32F);
    distCoeffs.at<float>(0, 0) = Config::k1(view);
    distCoeffs.at<float>(1, 0) = Config::k2(view);
    distCoeffs.at<float>(2, 0) = Config::p1(view);
    distCoeffs.at<float>(3, 0) = Config::p2(view);
    if (mvnDistCoeffs[view] == 5) {
        distCoeffs.at<float>(4, 0) = Config::k3(view);
    }
    distCoeffs.copyTo(mvDistCoeffs[view]);
}

std::ostream& operator<<(std::ostream& os, const Config& cfg)
{
    os << endl << "SLAM system parameters:" << endl;
    // camera parameters
    os << "### Camera Parameters ###" << endl;
    const vector<CameraParameters>& vCamParams = cfg.mvCamParams;
    const vector<int>& vnDistCoeffs = cfg.mvnDistCoeffs;
    for (unsigned i = 0; i < vCamParams.size(); ++i) {
        os << "- View " << i << ":" << endl
           << "  - Image resolution: "
           << vCamParams[i].w << "x" << vCamParams[i].h << endl
           << "  - FPS: " << vCamParams[i].fps << endl
           << "  - fx: " << vCamParams[i].fx << endl
           << "  - fy: " << vCamParams[i].fy << endl
           << "  - cx: " << vCamParams[i].cx << endl
           << "  - cy: " << vCamParams[i].cy << endl
           << "  - k1: " << vCamParams[i].k1 << endl
           << "  - k2: " << vCamParams[i].k2 << endl
           << "  - p1: " << vCamParams[i].p1 << endl
           << "  - p2: " << vCamParams[i].p2 << endl;
        if (vnDistCoeffs[i] == 5) {
            os << "  - k3: " << vCamParams[i].k3 << endl;
        }
    }
    // feature extraction parameters
    const FeatExtParameters& featParams = cfg.mFeatParams;
    os << "### Feature Extraction Parameters ###" << endl;
    os << "- Number of Features: " << featParams.nFeatures << endl
       << "- Scale Factor of Image Pyramid: " << featParams.scaleFactor << endl
       << "- Number of Pyramid Levels: " << featParams.nLevels << endl;
    return os;
}

} // namespace SLAM_demo
