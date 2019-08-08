/**
 * @file   System.hpp
 * @brief  Header of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <string>

namespace SLAM_demo {

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
     * @param[in] eMode System mode.
     */
    System(Mode eMode = Mode::MONOCULAR);
private:
    Mode meMode;
};

} // namespace SLAM_demo

#endif // SYSTEM_HPP
