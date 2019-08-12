/**
 * @file   Frame.hpp
 * @brief  Header of Frame class for storing intra- and inter-frame info.
 * @author Charlie Li
 * @date   2019.08.12
 */

#include "Frame.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp> // cv::imshow()
#include "Config.hpp"

using namespace std;
using namespace cv;

namespace SLAM_demo {

Frame::Frame(const Mat& img)
{
    // configure feature extractor
    const Config& cfg = Config::getInstance();
    mpFeatExtractor = ORB::create(Config::nFeatures(), Config::scaleFactor(),
                                  Config::nLevels());
    extractFeatures(img);
}

void Frame::extractFeatures(const cv::Mat& img)
{
    mpFeatExtractor->detectAndCompute(img, noArray(), mvKpts, mDesc);
}

} // namespace SLAM_demo
