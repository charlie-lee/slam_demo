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

using namespace std;
using namespace cv;

CamDataLoader::CamDataLoader(const string& strImgPath,
                             const string& strDataFile)
{
    ifstream ifs(strDataFile);
    string line;
    getline(ifs, line); // ignore the 1st row
    while (getline(ifs, line)) { // traverse data rows
        stringstream ss(line);
        string s;
        getline(ss, s, ','); // get timestamp info
        double t = stod(s) / 1e9; // timestamp: ns -> s conversion
        mvTimestamps.push_back(t);
        getline(ss, s); // get image filename
        mvstrImgs1.push_back(strImgPath + "/" + s);
    }
}

Mat CamDataLoader::loadImg(int nFrame, View eView)
{
    Mat img;
    if (eView == View::MONO || eView == View::LEFT || eView == View::RGB) {
        img = imread(mvstrImgs1[nFrame], IMREAD_UNCHANGED); // load to view 1
    } else if (eView == View::RIGHT || eView == View::DEPTH) {
        img = imread(mvstrImgs2[nFrame], IMREAD_UNCHANGED); // load to view 2
    }
    return img;
}