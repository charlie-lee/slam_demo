/**
 * @file   CamDataLoader.cpp
 * @brief  Camera data loader class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include "CamDataLoader.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

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

Mat CamDataLoader::loadImg(int nFrame, View view)
{
    Mat img;
    switch (view) {
    case View::MONO:
    case View::LEFT:
    case View::RGB:
        img = imread(mvstrImgs1[nFrame], IMREAD_UNCHANGED);
        break;
    case View::RIGHT:
    case View::DEPTH:
        img = imread(mvstrImgs2[nFrame], IMREAD_UNCHANGED);
        break;
    default:
        break;
    }
    return img;
}
