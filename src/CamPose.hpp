/**
 * @file   CamPose.hpp
 * @brief  Header of CamPose class for camera pose representation.
 * @author Charlie Li
 * @date   2019.08.23
 */

#ifndef CAMPOSE_HPP
#define CAMPOSE_HPP

#include <iostream>

#include <opencv2/core.hpp>
#include <Eigen/Core>

namespace SLAM_demo {

/**
 * @class CamPose
 * @brief Store camera pose of each frame.
 */
class CamPose {
public: // public members
    /// Store default pose and its inverse \f$[I|0]\f$.
    CamPose();
    /** 
     * @brief Construct pose-related data using the camera pose itself.
     * @param[in] Tcw \f$3 \times 4\f$ camera pose \f$[R_{cw}|t_{cw}]\f$,
     */
    CamPose(const cv::Mat& Tcw);
    /** 
     * @brief Construct pose-related data using camera rotation and 
     *        translation matrices.
     * @param[in] Rcw \f$3 \times 3\f$ camera rotation matrix.
     * @param[in] tcw \f$3 \times 1\f$ camera translation matrix.
     */
    CamPose(const cv::Mat& Rcw, const cv::Mat& tcw);
    /// Copy constructor.
    CamPose(const CamPose& pose);
    /// Copy-assignment operator.
    CamPose& operator=(const CamPose& pose);
    /// Default destructor.
    ~CamPose() = default;
    /// Set camera pose \f$[R|t]\f$ of current frame using an input pose.
    void setPose(const cv::Mat& Tcw);
    /** 
     * @brief Set camera pose \f$[R|t]\f$ of current frame using input 
     *        rotation and translation matrices.
     */
    void setPose(const cv::Mat& Rcw, const cv::Mat& tcw);
    /**
     * @name Getters for Pose-related Data
     * @brief A group of getters for retrieving pose-related data.
     * @return Matrix representation of pose-related data of type cv::Mat.
     */
    ///@{
    CamPose getCamPoseInv() const { return CamPose(getPoseInv()); }
    cv::Mat getPose() const { return mTcw; }
    cv::Mat getRotation() const { return mTcw.colRange(0, 3).rowRange(0, 3); }
    cv::Mat getRotationAngleAxis() const;
    cv::Mat getTranslation() const { return mTcw.rowRange(0, 3).col(3); }
    /** 
     * @brief Get \f$3 \times 3\f$ skew-symmetric matrix \f$[t]_x\f$ based on 
     *        \f$3 \times 1\f$ translation vector \f$t = (t_1, t_2, t_3)^T\f$.
     * 
     * \f[ [t]_x = \begin{bmatrix} 
     *               0 & -t_3 & t_2 \\ t_3 & 0 & -t_1 \\ -t_2 & t_1 & 0
     *             \end{bmatrix}
     * \f]
     */
    cv::Mat getTranslationSS() const;
    cv::Mat getPoseInv() const { return mTwc; }
    cv::Mat getRotationInv() const { return mTwc.rowRange(0, 3).colRange(0, 3); }
    //cv::Mat getRotationInvAngleAxis() const;
    /// Get \f$t_{wc}\f$, which is the camera origin in world coordinate system.
    cv::Mat getCamOrigin() const { return mTwc.rowRange(0, 3).col(3); }
    ///@}
    /**
     * @brief  Get Euler angle representation of rotation matrix \f$R_{cw}\f$
     *         as \f$3 \times 1\f$ vector \f$(yaw, pitch, roll)^T\f$. 
     *         Unit: degree.
     */
    Eigen::Matrix<float, 3, 1> getREulerAngleEigen() const;
    /**
     * @brief  Get quaternion representation of rotation matrix \f$R_{cw}\f$
     *         as \f$4 \times 1\f$ vector \f$(qw, qx, qy, qz)^T\f$. 
     */
    Eigen::Quaternion<float> getRQuatEigen() const;
    /// Get Quaternion representation of R^T.
    Eigen::Quaternion<float> getRInvQuatEigen() const;
    /// Pose multiplication & assignment.
    CamPose& operator*=(const CamPose& rhs);
    /// Pose multiplication.    
    CamPose operator*(const CamPose& rhs) const;
private: // private data
    /** 
     * @brief \f$3 \times 4\f$ camera pose \f$[R_{cw}|t_{cw}]\f$, i.e., the
     *        transformation from world to camera coordinate system.
     */
    cv::Mat mTcw;
    /** 
     * @brief \f$3 \times 4\f$ transformation matrix \f$[R_{wc}|t_{wc}]\f$
     *        from camera to world coordinate system. 
     *        \f$ T_{cw, 4 \times 4} = T_{wc, 4 \times 4}^{-1}\f$.
     */
    cv::Mat mTwc;
private: // private members
    /** 
     * @brief Update rotation matrix within \f$T_{cw}\f$, and at the same 
     *        time update corresponding inverse transformation \f$T_{wc}\f$.
     */
    void setRotation(const cv::Mat& Rcw);
    /** 
     * @brief Update translation matrix within \f$T_{cw}\f$, and at the same
     *        time update corresponding inverse transformation \f$T_{wc}\f$.
     */
    void setTranslation(const cv::Mat& tcw);
    /// Set inverse pose \f$T_{wc} = [R_{cw}^T | -R_{cw}^T t_{cw}]\f$.
    void setPoseInv();
    /// Get rotation matrix represented using Eigen.
    Eigen::Matrix<float, 3, 3> getRotationEigen() const;    
};

/// Display pose info.
std::ostream& operator<<(std::ostream& os, const CamPose& pose);

} // namespace SLAM_demo

#endif // CAMPOSE_HPP
