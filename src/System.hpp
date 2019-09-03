/**
 * @file   System.hpp
 * @brief  Header of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace SLAM_demo {

// forward declarations
class Tracker;
class Map;

/**
 * @class System
 * @brief Core SLAM system class. Control all sub-modules in the system.
 */
class System {
public: // public data
    /// SLAM system mode.
    enum class Mode {
        MONOCULAR,
        STEREO,
        RGBD
    };
    /// Index of current frame loaded into the SLAM system.
    static unsigned nCurrentFrame;
public: // public members
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
    void trackImgs(const std::vector<cv::Mat>& vImgs, double timestamp) const;
private: // private data
    Mode meMode; ///< System mode.
    std::shared_ptr<Tracker> mpTracker; ///< Pointer to tracker module.
    std::shared_ptr<Map> mpMap; ///< Pointer to the map.
};

} // namespace SLAM_demo

#endif // SYSTEM_HPP
