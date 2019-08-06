/**
 * @file   mono_vo.cpp
 * @brief  A monocular visual odometry system.
 * @author Charlie Li
 * @date   2019.08.05
 */

#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

/**
 * @class CamDataLoader
 * @brief Data loader class for camera image data and the corresponding 
 *        timestamp data.
 */
class CamDataLoader {
public:
    /** 
     * @brief Constructor for monocular case. 
     * @param[in]  strImgPath  Input path of images to be loaded.
     * @param[in]  strDataFile Input path of the csv data file containing
     *                         timestamp and image filename info.
     */
    CamDataLoader(const string &strImgPath, const string &strDataFile);
    /// Default destructor.
    ~CamDataLoader() = default;
    /// Different views for monocular/stereo/RGBD cases.
    enum class View {
        mono,  ///< The single view for monocular case.
        left,  ///< The left view for stereo case.
        right, ///< The right view for stereo case.
        RGB,   ///< The RGB view for RGBD case.
        D      ///< The D (depth) view for RGBD case.
    };
    /** 
     * @brief Load image data (1 view only) for monocular/stereo/RGBD cases.
     * @param[in] nFrame Frame index starting from 0.
     * @param[in] view   The image view to be loaded.
     * @return Loaded image data of cv::Mat type.
     */
    cv::Mat loadImg(int nFrame, View view = View::mono);
private:
    vector<string> mvstrImgs1; // image filenames (mono/left/RGB view)
    vector<string> mvstrImgs2; // image filenames (right/D view)
    vector<double> mvTimestamps; // timestamp info (unit: second)
};

CamDataLoader::CamDataLoader(const string &strImgPath,
                             const string &strDataFile)
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

cv::Mat CamDataLoader::loadImg(int nFrame, View view)
{
    cv::Mat img;
    switch (view) {
    case View::mono:
    case View::left:
    case View::RGB:
        break;
    case View::right:
    case View::D:
        break;
    default:
        break;
    }
    return img;
}


int main(int argc, char** argv)
{
    cout << "A Monocular Visual Odometry System." << endl;
    //CamDataLoader cdl(argv[1], argv[2]);
    return 0;
}
