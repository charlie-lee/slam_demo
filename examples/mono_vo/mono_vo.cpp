/**
 * @file   mono_vo.cpp
 * @brief  A monocular visual odometry system.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include "CamDataLoader.hpp"

#include "System.hpp"

#include <opencv2/highgui.hpp>

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    cout << "A Monocular Visual Odometry System." << endl;

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " imgPath camDataFile" << endl;
        return -1;
    }

    CamDataLoader cdl(argv[1], argv[2]);

    // temp test on data loading scheme
    cv::Mat img;
    for (int ni = 0; ni < cdl.getNFrames(); ni++) {
        img = cdl.loadImg(ni);
        cv::imshow("cam0", img);
        cv::waitKey(1);
    }
    
    return 0;
}
