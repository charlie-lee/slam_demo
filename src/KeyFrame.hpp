/**
 * @file   KeyFrame.hpp
 * @brief  Header of KeyFrame class for storing keyframe info.
 * @author Charlie Li
 * @date   2019.10.18
 */

#ifndef KEYFRAME_HPP
#define KEYFRAME_HPP

#include "FrameBase.hpp"

#include <map>
#include <memory>
#include <vector>

#include "CamPose.hpp"

namespace SLAM_demo {

// forward declarations
class Frame;
class MapPoint;

/**
 * @class KeyFrame
 * @brief Store keyframe info for local mapper and tracker.
 */
class KeyFrame : public FrameBase {
public: // public member functions
    /**
     * @brief Constructor of the KeyFrame class. Construct keyframe from
     *        normal frames.
     * @param[in] pFrame Pointer to the frame.
     */
    KeyFrame(const std::shared_ptr<Frame>& pFrame);
    /// Get frame index of current frame.
    unsigned index() const { return mnIdx; }
    /// Get keyframe index.
    unsigned frameIndex() const { return mnIdxFrame; }
    /// Update keyframe connections in the weighted graph.
    void updateConnections();
    /**
     * @brief Get all connected keyframes in the pose graph.
     * @return A vector of connected keyframes in random order.
     * @note The data is updated with the latest call to updateConnections().
     */
    std::vector<std::shared_ptr<KeyFrame>> getConnectedKFs() const;
    /**
     * @brief Get 'weak' keyframes that are not connected to the keyframe
     *        but have common map points with the keyframe.
     * @note The data is updated with the latest call to updateConnections().
     */
    std::vector<std::shared_ptr<KeyFrame>> getWeakKFs() const;
    /**
     * @brief Get specific number of connected keyframes in the pose graph.
     * @param[in] nKFs Number of requested keyframes.
     * @return A vector of connected keyframes sorted by weight in 
     *         decending order.
     * @note 
     * - If @p nKFs is too low (<= 0) or too high (> max), all 
     *   connected keyframes will be returned.
     * - The data is updated with the latest call to updateConnections().
     */
    std::vector<std::shared_ptr<KeyFrame>> getBestConnectedKFs(int nKFs) const;
    std::vector<std::shared_ptr<KeyFrame>> getRelatedKFs() const;
    /// Clear connection data of this KF and update that of all related KFs.
    void clearConnectionData();
private: // private member functions
    /// Do not allow copying.
    KeyFrame(const KeyFrame& rhs) = delete;
    /// Do not allow copy-assignment.
    KeyFrame& operator=(const KeyFrame& rhs) = delete;
private: // private members
    /// Minimum weight (common map points) between 2 keyframes.
    static const int TH_MIN_WEIGHT;
    unsigned mnIdx; ///< Keyframe index.
    unsigned mnIdxFrame; ///< Frame index.
    static unsigned nNextIdx; ///< Index for next keyframe.    
    std::vector<cv::KeyPoint> mvKpts; ///< Keypoint data of the current frame.
    /** 
     * @brief A map of (pKF, weight) pairs where weight is the number of 
     *        common keypoints between 2 keyframes.
     */
    std::map<std::shared_ptr<KeyFrame>, int> mmpKFnWt;
    /// A vector of keyframes that are not connected but have common map points.
    std::vector<std::shared_ptr<KeyFrame>> mvpWeakKFs;
};

} // namespace SLAM_demo

#endif // KEYFRAME_HPP
