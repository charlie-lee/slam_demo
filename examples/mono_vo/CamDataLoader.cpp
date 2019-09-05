/**
 * @file   CamDataLoader.cpp
 * @brief  Implementations of camera data loader class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include "CamDataLoader.hpp"

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using std::ifstream;
using std::stringstream;
using std::vector;
using std::string;
using cv::Mat;

CamDataLoader::CamDataLoader(const std::string& strImgPath,
                             const std::string& strDataFile,
                             double tsScale)
{
    ifstream ifs(strDataFile);
    string line;
    getline(ifs, line); // ignore the 1st row
    while (getline(ifs, line)) { // traverse data rows
        stringstream ss(line);
        string s;
        getline(ss, s, ','); // get timestamp info
        double t = stod(s) / tsScale; // timestamp: ns -> s conversion
        mvTimestamps.push_back(t);
        getline(ss, s); // get image filename
        mvstrImgs1.push_back(strImgPath + "/" + s);
    }
}

cv::Mat CamDataLoader::loadImg(int nFrame, View eView) const
{
    Mat img;
    // view 1 for monocular, left in stereo, and RGB in RGB-D
    // view 2 for right in stereo, and D in RGB-D
    if (eView == View::MONO || eView == View::LEFT || eView == View::RGB) {
        img = cv::imread(mvstrImgs1[nFrame], cv::IMREAD_UNCHANGED);
    } else if (eView == View::RIGHT || eView == View::DEPTH) {
        img = cv::imread(mvstrImgs2[nFrame], cv::IMREAD_UNCHANGED);
    }
    return img;
}
