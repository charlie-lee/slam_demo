/**
 * @file   LocalMapper.hpp
 * @brief  Header of local mapper class in SLAM system.
 * @author Charlie Li
 * @date   2019.10.22
 */

#ifndef LOCAL_MAPPER_HPP
#define LOCAL_MAPPER_HPP

#include <list>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace SLAM_demo {

// forward declarations
class KeyFrame;
class Map;
class MapPoint;
class Optimizer;

/**
 * @class LocalMapper
 * @brief Construct local map and triangulate new map points.
 *
 * Currently the procedure in the local mapper is as follows:
 * 1. Process each newly added keyframe:
 *    - Create new map points between it & a number of its connected keyframes;
 *    - Fuse newly triangulated map points to the related keyframes in the local
 *      map (cosivibility graph);
 *    - Perform local bundle adjustment (based on local map of the keyframe).
 *    - Remove redundant keyframes.
 * 2. Remove redundant map points.
 */
class LocalMapper {
public: // public members
    /// Constructor of the class.
    LocalMapper(const std::shared_ptr<Map>& pMap,
                const std::shared_ptr<Optimizer>& pOptimizer);
    /// Core function for all the related tasks.
    void run();
    /// Insert a keyframe to the back of the list of new keyframes.
    void insertKeyFrame(const std::shared_ptr<KeyFrame>& pKF);
private: // private data
    /// Number of best connected keyframes for new map point creation.
    static const int NUM_BEST_KF;
    std::shared_ptr<Map> mpMap; ///< Pointer to the map.
    std::shared_ptr<Optimizer> mpOpt; ///< Pointer to the optimizer.
    /// List of newly inserted keyframes.
    std::list<std::shared_ptr<KeyFrame>> mlNewKFs;
    /// Current keyframe to be processed.
    std::shared_ptr<KeyFrame> mpKFCur;
private: // private members
    /// Create new map points between current and its connected KFs.
    void createNewMapPoints() const;
    /// Fuse new map points to the local map of newly added keyframe.
    void fuseNewMapPoints() const;
    /// Remove redundant keyframes from the map.
    void removeKeyFrames() const;
    /** 
     * @brief Fuse newly triangulated points to the 2 keyframes which
     *        triangulate the points.
     * @param[in] pKF2       Pointer to keyframe 2 (newly added keyframe).
     * @param[in] pKF1       Pointer to keyframe 1 (keyframe in the map).
     * @param[in] vMatches21 2D-to-2D matches between keyframe 2 (querying set) 
     *                       and 1 (training set).
     * @param[in] vXws       A vector of newly triangulated points (same size 
     *                       with that of the matches).
     * @return Number of newly created map points.
     * @note Newly triangulated points which can be bound to the input keyframe
     *       are generated as new map points and are put into the map.
     */
    int fuseMapPoints(const std::shared_ptr<KeyFrame>& pKF2,
                      const std::shared_ptr<KeyFrame>& pKF1,
                      const std::vector<cv::DMatch>& vMatches21,
                      const std::vector<cv::Mat>& vXws) const;
    /** 
     * @brief Fuse map points of source keyframe to destination keyframe.
     * @param[in] pKFsrc Pointer to source keyframe.
     * @param[in] pKFdst Pointer to destination keyframe.
     */
    void fuseMapPoints(const std::shared_ptr<KeyFrame>& pKFsrc,
                       const std::shared_ptr<KeyFrame>& pKFdst) const;
    /**
     * @brief Check if a newly triangulated 3D point is better than
     *        already-bound map points generated from the same keypoint
     *        of both keyframes.
     * @param[in] Xw      Newly triangulated \f$3 \times 1\f$ point in world 
     *                    coordinate.
     * @param[in] pKF2    Pointer to keyframe 2 (newly added keyframe).
     * @param[in] pKF1    Pointer to keyframe 1 (connected keyframe).
     * @param[in] match21 Match data between @p pKF2 and @p pKF1.
     * @return True if the new point is better or there's no bound map point.
     */
    bool isNewPtBetter(const cv::Mat& Xw,
                       const std::shared_ptr<KeyFrame>& pKF2,
                       const std::shared_ptr<KeyFrame>& pKF1,
                       const cv::DMatch& match21) const;
};

} // namespace SLAM_demo

#endif // LOCAL_MAPPER_HPP
