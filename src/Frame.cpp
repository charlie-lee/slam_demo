/**
 * @file   Frame.hpp
 * @brief  Header of Frame class for storing intra- and inter-frame info.
 * @author Charlie Li
 * @date   2019.08.12
 */

#include "Frame.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "Config.hpp"

namespace SLAM_demo {

using cv::Mat;
using cv::ORB;

Frame::Frame(const Mat& img)
{
    // configure feature extractor
    const Config& cfg = Config::getInstance();
    mpFeatExtractor = ORB::create(Config::nFeatures(), Config::scaleFactor(),
                                  Config::nLevels());
    extractFeatures(img);
}

void Frame::extractFeatures(const Mat& img)
{
    mpFeatExtractor->detectAndCompute(img, cv::noArray(), mvKpts, mDesc);
}

} // namespace SLAM_demo
