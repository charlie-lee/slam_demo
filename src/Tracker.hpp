/**
 * @file   Tracker.hpp
 * @brief  Header of tracker class in SLAM system.
 * @author Charlie Li
 * @date   2019.08.09
 */

#ifndef TRACKER_HPP
#define TRACKER_HPP

#include "System.hpp"

namespace SLAM_demo {

/**
 * @class Tracker
 * @brief Track adjacent 2 frames for feature extraction and matching.
 */
class Tracker {
public:
    /**
     * @brief Constructor.
     * @param[in] eMode See System::Mode for more information.
     */
    Tracker(System::Mode eMode);
private:
    System::Mode meMode;
};

} // namespace SLAM_demo

#endif // TRACKER_HPP
