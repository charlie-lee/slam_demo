/**
 * @file   FeatureMatcher.hpp
 * @brief  Header of feature matcher class in SLAM system.
 * @author Charlie Li
 * @date   2019.10.22
 */

#ifndef FEATURE_MATCHER_HPP
#define FEATURE_MATCHER_HPP

//#include <map> // temp
#include <memory>
#include <vector>

#include <opencv2/features2d.hpp>
#include "MapPoint.hpp" // temp

namespace SLAM_demo {

// forward declarations
class FrameBase;

/**
 * @class FeatureMatcher
 * @brief Provide feature matching operations between 2D/3D points.
 */
class FeatureMatcher {
public: // public members
    /**
     * @brief Constructor of the class.
     * @param[in] thDistMatchMax    Max pixel distance between 2 matched points.
     * @param[in] bUseLoweRatioTest Whether to use Lowe's ratio test.
     * @param[in] thRatioTest       Max threshold of the ratio of smallest 
     *                              feature distance to that of 2nd smallest.
     * @param[in] thAngMatchMax     Max angle difference between 2 matched 
     *                              points (in degrees).
     * @param[in] thDistDescMax     Max feature distance (Hamming) between 2
     *                              descriptors.
     */
    FeatureMatcher(float thDistMatchMax, bool bUseLoweRatioTest = true,
                   float thRatioTest = 0.7f, float thAngMatchMax = 15.0f,
                   int thDistDescMax = 15);
    /** 
     * @brief Match features for 2D-to-2D case between 2 frames.
     * @return A vector of matching keypoints of cv::DMatch type.
     */
    std::vector<cv::DMatch> match2Dto2D(
        const std::shared_ptr<FrameBase>& pF2,
        const std::shared_ptr<FrameBase>& pF1) const;
    /** 
     * @brief Match features for 2D-to-2D case between 2 frames
     *        (custom implementation).
     *
     * The custom implementation will try to get a minimum set of features
     * for each feature to be matched based on pixel distance and orientation.
     *
     * @return A vector of matching keypoints of cv::DMatch type.
     */
    std::vector<cv::DMatch> match2Dto2DCustom(
        const std::shared_ptr<FrameBase>& pF2,
        const std::shared_ptr<FrameBase>& pF1) const;
    /** 
     * @brief Match features for 2D-to-3D case between 2 frames.
     * @param[in] pF2       Pointer to frame 2 which has features for 2D points.
     * @param[in] pF1       Pointer to frame 1 which has features for 3D points.
     * @param[in] bBindMPts Whether to bind newly matched map points to @p pF2.
     * @return A vector of matching keypoints of cv::DMatch type, where
     *         keypoints in frame 2 are in the querying set, and those
     *         in frame 1 are in the training set.
     */
    std::vector<cv::DMatch> match2Dto3D(
        const std::shared_ptr<FrameBase>& pF2,
        const std::shared_ptr<FrameBase>& pF1,
        bool bBindMPts = true) const;
    /** 
     * @brief Match features for 2D-to-3D case between 2 frames.
     * @param[in] pF2    Pointer to frame 2 which has features for 2D points.
     * @param[in] vpMPts A std::vector of pointers to 3D map points.
     * @param[in] bBindMPts Whether to bind newly matched map points to @p pF2.
     * @return A vector of matching keypoints of cv::DMatch type, where
     *         keypoints in frame 2 are in the querying set, and those
     *         in @p vpMPts are in the training set.
     */
    std::vector<cv::DMatch> match2Dto3D(
        const std::shared_ptr<FrameBase>& pF2,
        const std::vector<std::shared_ptr<MapPoint>>& vpMPts,
        bool bBindMPts = true) const;
    /** 
     * @brief Match features for 2D-to-3D case between 2 frames 
     *        (custom implementation).
     *
     * The custom implementation will try to get a minimum set of features
     * for each feature to be matched based on pixel distance and orientation.
     *
     * @param[in] pF2       Pointer to frame 2 which has features for 2D points.
     * @param[in] pF1       Pointer to frame 1 which has features for 3D points.
     * @param[in] bBindMPts Whether to bind newly matched map points to @p pF2.
     * @return A vector of matching keypoints of cv::DMatch type, where
     *         keypoints in frame 2 are in the querying set, and those
     *         in frame 1 are in the training set.
     */
    std::vector<cv::DMatch> match2Dto3DCustom(
        const std::shared_ptr<FrameBase>& pF2,
        const std::shared_ptr<FrameBase>& pF1,
        bool bBindMPts = true) const;
    /** 
     * @brief Match features for 2D-to-3D case between 2 frames
     *        (custom implementation).
     *
     * The custom implementation will try to get a minimum set of features
     * for each feature to be matched based on pixel distance and orientation.
     *
     * @param[in] pF2    Pointer to frame 2 which has features for 2D points.
     * @param[in] vpMPts A std::vector of pointers to 3D map points.
     * @param[in] bBindMPts Whether to bind newly matched map points to @p pF2.
     * @return A vector of matching keypoints of cv::DMatch type, where
     *         keypoints in frame 2 are in the querying set, and those
     *         in @p vpMPts are in the training set.
     */
    std::vector<cv::DMatch> match2Dto3DCustom(
        const std::shared_ptr<FrameBase>& pF2,
        const std::vector<std::shared_ptr<MapPoint>>& vpMPts,
        bool bBindMPts = true) const;
private: // private data
    /// Feature matcher.
    std::shared_ptr<cv::DescriptorMatcher> mpFeatMatcher;
    float mThDistMatchMax; ///< Max pixel distance between 2 matched points.
    bool mbUseLoweRatioTest; ///< Whether to use Lowe's ratio test.
    /// Max ratio threshold of smallest feature distance to that of 2nd smallest.
    float mThRatioTest;
    float mThAngMatchMax; ///< Max angle difference between 2 matched keypoints.
    int mThDistDescMax; ///< Max descriptor distance between 2 descriptors.
private: // private members
    /**
     * @brief Filter valid feature matching result.
     * @param[in] vvMatches21 A vector of vector of cv::DMatch matches.
     * @param[in] pF2         Pointer to frame 2.
     * @param[in] pF1         Pointer to frame 1 which has features for 3D pts.
     * @param[in] vbMask1     Mask for keypoints in frame 1 for 2D-to-3D case.
     * @return A vector of matching keypoints of cv::DMatch type, where
     *         keypoints in frame 2 are in the querying set, and those
     *         in frame 1 are in the training set.
     * @note @p vbMask1[i] is true if \f$i\f$th keypoint is bound with a valid 
     *       map point.
     */
    std::vector<cv::DMatch> filterMatchResult(
        const std::vector<std::vector<cv::DMatch>>& vvMatches21,
        const std::shared_ptr<FrameBase>& pF2,
        const std::shared_ptr<FrameBase>& pF1,
        const std::vector<bool>& vbMask1 = std::vector<bool>()) const;
    /**
     * @brief Filter valid feature matching result.
     * @param[in] vvMatches21 A vector of vector of cv::DMatch matches.
     * @param[in] pF2         Pointer to frame 2.
     * @param[in] vpMPts      A std::vector of pointers to 3D map points.
     * @return A vector of matching keypoints of cv::DMatch type, where
     *         keypoints in frame 2 are in the querying set, and those
     *         in frame 1 are in the training set.
     */
    std::vector<cv::DMatch> filterMatchResult(
        const std::vector<std::vector<cv::DMatch>>& vvMatches21,
        const std::shared_ptr<FrameBase>& pF2,
        const std::vector<std::shared_ptr<MapPoint>>& vpMPts) const;
    /** 
     * @brief Get mask info for 2D-to-3D matching.
     *
     * The mask specifies which query and training descriptors can be
     * matched. Namely, queryDescriptors[i] can be matched with 
     * trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero.
     *
     * @param[in] vIdxKpts1 Indices of keypoints bound with valid map points.
     * @param[in] nKpts2    Number of keypoints in frame 2.
     * @param[in] nKpts1    Number of keypoints in frame 1.
     * @return Mask data described above.
     * @note knnMatch() seems not using provided mask! Currently 
     *       disable this function.
     */
    cv::Mat getMatchMask2Dto3D(const std::vector<int>& vIdxKpts1,
                               int nKpts2, int nKpts1) const;
    /**
     * @brief Compute Hamming distance between 2 256-bit descriptors.
     * @param[in] a 1st feature descriptor.
     * @param[in] b 2nd feature descriptor.
     * @return Hamming distance between descriptor @p a and @p b.
     * @see http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel 
     *      for the bit set count operation implemented in this function.
     */
    int hammingDistance(const cv::Mat& a, const cv::Mat& b) const;
};

} // namespace SLAM_demo

#endif // FEATURE_MATCHER_HPP
