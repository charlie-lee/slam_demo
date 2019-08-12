/**
 * @file   System.hpp
 * @brief  Header of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <string>
#include <memory>

#include <opencv2/core.hpp>

namespace SLAM_demo {

// forward declarations
class Tracker;

/**
 * @class System
 * @brief Core SLAM system class. Control all sub-modules in the system.
 */
class System {
public:
    /// SLAM system mode
    enum class Mode {
        MONOCULAR,
        STEREO,
        RGBD
    };
    /**
     * @brief Constructor for SLAM system class,
     * @param[in] eMode SLAM system mode (see System::Mode for details).
     */
    System(Mode eMode);
    /**
     * @brief Track input images.
     * @param[in] vImgs     A vector of input images captured with the same
     *                      timestamp info.
     * @param[in] timestamp The timestamp info.
     */
    void trackImgs(const std::vector<cv::Mat>& vImgs, double timestamp);
private: // private data
    Mode meMode;
    std::shared_ptr<Tracker> mpTracker; // pointer to tracker module
};

} // namespace SLAM_demo

#endif // SYSTEM_HPP
