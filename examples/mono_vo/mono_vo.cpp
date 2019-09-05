/**
 * @file   mono_vo.cpp
 * @brief  A monocular visual odometry system.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include "Config.hpp"
#include "System.hpp"
#include "CamDataLoader.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

/**
 * @brief Main function for the monocular VO application.
 * @param[in] argc Number of arguments.
 * @param[in] argv Set of input arguments:
 *                 - @p argv[1]: path to the system configuration file;
 *                 - @p argv[2]: path to the input images captured by a camera;
 *                 - @p argv[3]: path to the camera data file containing 
 *                               timestamp and image filename info.
 * @return Exit value. 0 if the system runs successfully.
 */
int main(int argc, char** argv)
{
    cout << "A Monocular Visual Odometry System." << endl;

    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " cfgFile imgPath camDataFile [timestampScale]" << endl;
        return 1;
    }

    // set and display global parameters for the SLAM system
    SLAM_demo::Config& cfg = SLAM_demo::Config::getInstance();
    if (!cfg.setParameters(argv[1], SLAM_demo::System::Mode::MONOCULAR)) {
        cerr << "SLAM system configuration failed." << endl;
        return 1;
    }
    cout << cfg << endl;

    // initialize the SLAM system
    SLAM_demo::System SLAM(SLAM_demo::System::Mode::MONOCULAR);

    // parse and load camera-captured data
    double tsScale = 1e9; // timestamp scale for conversion to unit of second
    if (argc == 5) {
        std::string stsScale(argv[4]);
        tsScale = stod(stsScale);
    }
    CamDataLoader cdl(argv[2], argv[3], tsScale);

    // load each frame and its corresponding timestamp into the SLAM system
    for (int ni = 0; ni < cdl.getNFrames(); ni++) {
        cv::Mat img;
        img = cdl.loadImg(ni, CamDataLoader::View::MONO);
        assert(img.cols == SLAM_demo::Config::width() &&
               img.rows == SLAM_demo::Config::height());
        // load a vector of images into the SLAM system
        vector<cv::Mat> vImgs;
        vImgs.reserve(1);
        vImgs.push_back(img);
        SLAM.trackImgs(vImgs, cdl.getTimestamp(ni));
    }
    // dump SLAM results
    SLAM.dumpTrajectory();
    return 0;
}
