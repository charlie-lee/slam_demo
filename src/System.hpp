/**
 * @file   System.hpp
 * @brief  Header of core SLAM system class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class LocalMapper;
class Map;
class Optimizer;
class Tracker;

/**
 * @class System
 * @brief Core SLAM system class. Control all sub-modules in the system.
 */
class System {
public: // public members
    /// SLAM system mode.
    enum class Mode {
        MONOCULAR,
        STEREO,
        RGBD,
    };
    /// Trajectory dump mode.
    enum class DumpMode {
        REAL_TIME, ///< Real time trajectory.
        OPTIMIZED, ///< Optimized trajectory.
    };
    /// Index of current frame loaded into the SLAM system.
    static unsigned nCurrentFrame;
    /// Number of consecutively lost frames.
    static unsigned nLostFrames;
    /// Maxisum number of lost frames allowed in the system.
    const static unsigned TH_MAX_LOST_FRAMES;    
public: // public member functions
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
    /**
     * @name Dump trajectory data to disk.
     *
     * For TUM dataset, the format of each pose is as follows:
     * timestamp tx ty tz qx qy qz qw
     *
     * @param[in] eMode Dump mode defined in System::DumpMode.
     *
     * @note Default filename: trajectoryRT.txt for DumpMode::REAL_TIME,
     *       and trajectoryOpt.txt for DumpMode::OPTIMIZED.
     */
    void dumpTrajectory(DumpMode eMode);
    /// System reset.
    void reset();    
private: // private members
    Mode meMode; ///< System mode.
    std::shared_ptr<Map> mpMap; ///< Pointer to the map.
    std::shared_ptr<Optimizer> mpOptimizer; ///< Pointer to the optimizer.
    std::shared_ptr<LocalMapper> mpLocalMapper; ///< Pointer to local mapper.
    std::shared_ptr<Tracker> mpTracker; ///< Pointer to tracker module.
    /// Real-time trajectory of the system.
    std::map<double, CamPose> mmTrajectoryRT;
    /// Optimized trajectory of the system.
    std::map<double, CamPose> mmTrajectoryOpt;
private: // private member functions
    /**
     * @brief Save real time trajectory info of 1 timestamp to the system.
     * @param[in] timestamp The timestamp info.
     * @param[in] pose      Camera pose at the corresponding timestamp.
     */
    void saveTrajectoryRT(double timestamp, const CamPose& pose);
    /// Save optimized trajectory info to the system.
    void saveTrajectoryOpt();
};

} // namespace SLAM_demo

#endif // SYSTEM_HPP
