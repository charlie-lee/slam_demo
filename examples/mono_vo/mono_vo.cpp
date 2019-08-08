/**
 * @file   mono_vo.cpp
 * @brief  A monocular visual odometry system.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include <iostream>
#include <memory> // std::unique_ptr

#include <opencv2/highgui.hpp>
#include "Config.hpp"
#include "System.hpp"
#include "CamDataLoader.hpp"

using namespace std;

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

    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " cfgFile imgPath camDataFile" << endl;
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
    SLAM_demo::System(SLAM_demo::System::Mode::MONOCULAR);

    // parse and load camera-captured data
    CamDataLoader cdl(argv[2], argv[3]);

    // temp test on data loading scheme
    cv::Mat img;
    for (int ni = 0; ni < cdl.getNFrames(); ni++) {
        img = cdl.loadImg(ni, CamDataLoader::View::MONO);
        cv::imshow("cam0", img);
        cv::waitKey(1);
    }
    
    return 0;
}
