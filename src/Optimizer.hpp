/**
 * @file   Optimizer.hpp
 * @brief  Header of Optimizer class for pose & map data optimization.
 * @author Charlie Li
 * @date   2019.09.17
 */

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <memory>

#include <opencv2/core.hpp>
#include <g2o/types/slam3d/types_slam3d.h>

namespace SLAM_demo {

// forward declarations
class Frame;
class Map;

/**
 * @class Optimizer
 * @brief Optimize pose & map data by bundle adjustment.
 */
class Optimizer {
public: // public members
    /// Constructor.
    Optimizer(const std::shared_ptr<Map>& pMap);
    /**
     * @brief Global bundle adjustment for pose & map data optimization.
     * @param[in] nFrames Number of recent frames as input to global BA scheme.
     *                    Optimize all avaiable frames if nFrames == 0.
     * @param[in] nIter   Number of iterations for the optimization scheme.
     * @param[in] bRobust Whether or not to use robust kernel.
     */
    void globalBundleAdjustment(unsigned nFrames = 0, int nIter = 20,
                                bool bRobust = true) const;
    /**
     * @brief Single-frame bundle adjustment for pose & new map point 
     *        data optimization.
     * @param[in] nIter   Number of iterations for the optimization scheme.
     * @param[in] bRobust Whether or not to use robust kernel.
     * @note Only the new pose and all observed map points are being
     *       optimized. If there're no new triangulated map points avaible,
     *       the optimization scheme will be skipped.
     */
    void frameBundleAdjustment(unsigned nFrames = 1, int nIter = 20,
                               bool bRobust = true) const;
    /**
     * @brief Pose optimization scheme using BA.
     * @param[in] pFrame Pointer to the frame where the corresponding pose 
     *                   will be optimized.
     * @return Number of map point inliers after the optimization.
     * @note In this BA scheme, only the pose is optimized.
     */
    int poseOptimization(const std::shared_ptr<Frame>& pFrame) const;
private: // private data
    /// Pointer to the map.
    std::shared_ptr<Map> mpMap;
private: // private members
    /**
     * @name Conversion functions between cv::Mat & g2o vertices data
     */
    ///@{
    /** 
     * @brief Conversion from cv::Mat to g2o::SE3Quat.
     * @note Data precision: float -> double.
     */
    g2o::SE3Quat cvMat2SE3Quat(const cv::Mat& Tcw) const;
    /** 
     * @brief Conversion from g2o::SE3Quat to cv::Mat (3x4).
     * @note Data precision: double -> float.
     */
    cv::Mat SE3Quat2cvMat(const g2o::SE3Quat& T) const;
    /** 
     * @brief Conversion from cv::Mat (3x1) to Eigen::Vector3d.
     * @note Data precision: float -> double.
     */
    Eigen::Vector3d cvMat2Vector3d(const cv::Mat& X3D) const;
    /** 
     * @brief Conversion from Eigen::Vector3d to cv::Mat (3x1).
     * @note Data precision: double -> float.
     */
    cv::Mat Vector3d2cvMat(const Eigen::Vector3d& X) const;
    ///@}
};

} // namespace SLAM_demo

#endif // OPTIMIZER_HPP
