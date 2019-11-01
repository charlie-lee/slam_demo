/**
 * @file   Utility.hpp
 * @brief  Header of utility class in SLAM system for general functions.
 * @author Charlie Li
 * @date   2019.10.25
 */

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace SLAM_demo {

// forward declarations
class CamPose;
class FrameBase;

/** 
 * @class Utility
 * @brief Some general operations.
 */
class Utility {
public: // public data
    /// Cosine of smallest appropriate parallax/angle between 2 views.
    static const float TH_COS_PARALLAX;
    /// Reprojection error computation scheme.
    enum class ReprojErrScheme {
        F, ///< Fundamental matrix as reprojection transformation.
        H, ///< Homography as reprojection transformation.
    };
public: // public members
    Utility() = default;
    /** 
     * @brief Check whether a \f$2 \times 1\f$ cv::Mat point is inside 
     *        the border of an undistorted image.
     * @param[in] pt \f$2 \times 1\f$ point of cv::Mat type.
     * @return True if the 2D point is inside the border.
     */
    static bool is2DPtInBorder(const cv::Mat& pt);
    /** 
     * @brief Triangulate 3D points in world coordinate frame.
     * @param[in] pF2        Pointer to frame 2 (newer, querying set).
     * @param[in] pF1        Pointer to frame 1 (older, training set).
     * @param[in] vMatches21 2D-to-2D matching result between KF2 and KF1.
     * @return A std::vector of cv::Mat \f$3 \times 1\f$ triangulated points.
     * @note The returned std::vector should have same size with @p vMatches21.
     */
    static std::vector<cv::Mat> triangulate3DPts(
        const std::shared_ptr<FrameBase>& pF2,
        const std::shared_ptr<FrameBase>& pF1,
        const std::vector<cv::DMatch>& vMatches21);
    /**
     * @brief Check whether a triangulated 3D world point is good enough
     *        to be included to the map.
     *
     * Basically there're 5 things to check:
     * - The depth of the 3D camera coordinates in both views (whether the depth
     *   is negative)
     * - The parallax of the 2 views (the angle oP-Xw-oC where oP and oC are 
     *   the camera origins in previous and current frame, Xw is the 
     *   triangulated point) (whether the parallax is too low),
     * - The position of reprojected 2D image point (whether it is outside the 
     *   image border);
     * - The reprojection errors in both views (whether it is too high).
     * - Epipolar constriant for the keypoint pair.
     *
     * @param[in] Xw    \f$3 \times 1\f$ triangulated 3D world point in
     *                  inhomogeneous coordinate.
     * @param[in] kpt1  2D keypoint in previous frame.
     * @param[in] kpt2  corresponding 2D keypoint in current frame.
     * @param[in] pose1 Camera pose for view 1.
     * @param[in] pose2 Camera pose for view 2.
     * @return True if the triangulated point is good enough, otherwise false.
     */
    static bool checkTriangulatedPt(const cv::Mat& Xw,
                                    const cv::KeyPoint& kpt1,
                                    const cv::KeyPoint& kpt2,
                                    const CamPose& pose1,
                                    const CamPose& pose2,
                                    float thCosParallax = 0.9999f);
    /** 
     * @brief Compute fundamental matrix F21 of 2 views (from view 1 to view 2)
     *        based on their relative  poses T1 & T2.
     * @param[in] CP1 Camera pose of the 1st view.
     * @param[in] CP2 Camera pose of the 2nd view.
     * @return Fundamental matrix F21.
     */
    static cv::Mat computeFundamental(const CamPose& CP1, const CamPose& CP2);
    /**
     * @brief Compute reprojection error based on transformation matrix
     *        @p T21 (reproject @p p1 from view 1 to view 2) and @p T12
     *        (reproject @p p2 from view 2 to view 1).
     *
     * Given 2D-to-2D match \f$p_1 = (x_1, y_1, 1)\f$, 
     * \f$p_2 = (x_2, y_2, 1)\f$, and transformation matrix \f$T_{21}\f$,
     * \f$T_{12}\f$,
     * 
     * Reprojection error \f$e_F\f$ for \f$F\f$ where \f$T_{21} = F_{21}\f$ and
     * \f$T_{12} = F_{12} = F_{21}^T\f$:
     * \f{align}{
     *   e_F &= d(p_2, F_{21} p_1)^2 + d(p_1, F_{12} p_2)^2 \\
     *       &= d(p_2, l_2)^2 + d(p_1, l_1)^2 \\
     *       &= \frac{a_2 x_2 + b_2 y_2 + c_2}{a_2^2 + b_2^2} + 
     *          \frac{a_1 x_1 + b_1 y_1 + c_1}{a_1^2 + b_1^2}
     * \f}
     * where \f$l_1 = (a_1, b_1, c_1)\f$ and \f$l_2 = (a_2, b_2, c_2)\f$ are
     * the epipolar lines of triangulated point \f$P\f$ based on \f$p_1\f$, 
     * \f$p_2\f$ in view 1 and 2, respectively.
     *
     * Reprojection error \f$e_H\f$ for \f$H\f$ where \f$T_{21} = H_{21}\f$ and
     * \f$T_{12} = H_{12} = H_{21}^{-1}\f$:
     * \f{align}{
     *   e_H &= d(p_2, H_{21} p_1)^2 + d(p_1, H_{12} p_2)^2 \\
     *       &= d(p_2, p_2')^2 + d(p_1, p_1')^2 \\
     *       &= |x_2 - x_2'|^2 + |y_2 - y_2'|^2 + 
     *          |x_1 - x_1'|^2 + |y_1 - y_1'|^2
     * \f}
     * where \f$p_1' = (x_1', y_1')\f$, \f$p_2' = (x_2', y_2')\f$.
     *
     * @param[in] T21     Transformation matrix from view 1 to view 2.
     * @param[in] T12     Transformation matrix from view 2 to view 1.
     * @param[in] p1      Point in view 1.
     * @param[in] p2      Point in view 2.
     * @param[in] eScheme Computation scheme based on different transformation
     *                    matrices (see Tracker::ReprojErrScheme for details).
     * @return Reprojection error (square form) in FP32 precision.
     */
    static float computeReprojErr(const cv::Mat& T21, const cv::Mat& T12,
                                  const cv::Mat& p1, const cv::Mat& p2,
                                  ReprojErrScheme eScheme);
};

} // namespace SLAM_demo

#endif // UTILITY_HPP
