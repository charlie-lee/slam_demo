/**
 * @file   mono_vo.cpp
 * @brief  A monocular visual odometry system.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include <chrono> // timer-related
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include "Config.hpp"
#include "System.hpp"
#include "CamDataLoader.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
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
    shared_ptr<SLAM_demo::System> pSLAM = make_shared<SLAM_demo::System>(
        SLAM_demo::System::Mode::MONOCULAR);

    // parse and load camera-captured data
    double tsScale = 1e9; // timestamp scale for conversion to unit of second
    if (argc == 5) {
        std::string stsScale(argv[4]);
        tsScale = stod(stsScale);
    }
    CamDataLoader cdl(argv[2], argv[3], tsScale);
    int nFrames = cdl.getNFrames();

    // load each frame and its corresponding timestamp into the SLAM system
    std::chrono::duration<double> duration =
        std::chrono::duration<double>::zero();
    for (int ni = 0; ni < nFrames; ni++) {
        cv::Mat img;
        img = cdl.loadImg(ni, CamDataLoader::View::MONO);
        assert(img.cols == SLAM_demo::Config::width() &&
               img.rows == SLAM_demo::Config::height());
        // load a vector of images into the SLAM system
        vector<cv::Mat> vImgs;
        vImgs.reserve(1);
        vImgs.push_back(img);
        
        // invoke SLAM system and compute FPS
        auto start = std::chrono::high_resolution_clock::now();
        pSLAM->trackImgs(vImgs, cdl.getTimestamp(ni));
        auto end = std::chrono::high_resolution_clock::now();
        duration += end - start;
    }
    // dump SLAM results (both real-time and final-optimized)
    pSLAM->dumpTrajectory(SLAM_demo::System::DumpMode::REAL_TIME);
    pSLAM->dumpTrajectory(SLAM_demo::System::DumpMode::OPTIMIZED);
    
    // display FPS data
    cout << "FPS: " << nFrames / duration.count() << endl;
    return 0;
}
