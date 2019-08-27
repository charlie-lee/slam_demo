/**
 * @file   Config.hpp
 * @brief  Header of SLAM system configuration class.
 * @author Charlie Li
 * @date   2019.08.07
 */

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <iostream>
#include <memory> // std::unique_ptr
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include "System.hpp"

namespace SLAM_demo {

/// Camera parameters.
struct CameraParameters {
    int w;     ///< Width of captured image.
    int h;     ///< Height of captured image.
    float fps; ///< Camera framerate.
    float fx;  ///< Focal length (x-axis).
    float fy;  ///< Focal length (y-axis).
    float cx;  ///< Camera center \f$x_c\f$ in pixel.
    float cy;  ///< Camera center \f$y_c\f$ in pixel.
    float k1;  ///< Camera distortion coefficient \f$k_1\f$.
    float k2;  ///< Camera distortion coefficient \f$k_2\f$.
    float p1;  ///< Camera distortion coefficient \f$p_1\f$.
    float p2;  ///< Camera distortion coefficient \f$p_2\f$.
    float k3;  ///< Camera distortion coefficient \f$k_3\f$.
};

/// Feature extraction parameters.
struct FeatExtParameters {
    int nFeatures;     ///< Number of extracted features.
    float scaleFactor; ///< Scale factor of the image pyramid.
    int nLevels;       ///< Number of octaves/levels of the image pyramid.
};

/**
 * @class Config
 * @brief The class stores SLAM system configurations using singleton 
 *        design pattern.
 */
class Config {
public:
    friend std::ostream& operator<<(std::ostream& os, const Config& cfg);
    /// Get the only instance of the class.
    static Config& getInstance()
    {
        static std::unique_ptr<Config> instance(new Config);
        return *instance; // dereferencing to get the reference
    }
    // setters
    /**
     * @brief Set system parameters from a config file.
     *
     * The function will use cv::FileStorage class 
     * for parameter parsing on a YAML file.
     *
     * @param[in] strCfgFile Filename of the configuration file.
     * @param[in] eMode      SLAM system mode (see System::Mode for details).
     * @return A boolean value indicating if the configuration is successful.
     */
    bool setParameters(const std::string& strCfgFile,
                       System::Mode eMode);
    // getters
    /** 
     * @name Getters for Camera Parameters
     * @brief A group of camera parameter getters.
     * @param[in] i Get the parameter from the \f$i\f$th camera.
     */
    ///@{
    static int width(int i = 0) { return getInstance().mvCamParams[i].w; }
    static int height(int i = 0) { return getInstance().mvCamParams[i].h; }
    static float fps(int i = 0) { return getInstance().mvCamParams[i].fps; }
    static float fx(int i = 0) { return getInstance().mvCamParams[i].fx; }
    static float fy(int i = 0) { return getInstance().mvCamParams[i].fy; }
    static float cx(int i = 0) { return getInstance().mvCamParams[i].cx; }
    static float cy(int i = 0) { return getInstance().mvCamParams[i].cy; }
    static float k1(int i = 0) { return getInstance().mvCamParams[i].k1; }
    static float k2(int i = 0) { return getInstance().mvCamParams[i].k2; }
    static float p1(int i = 0) { return getInstance().mvCamParams[i].p1; }
    static float p2(int i = 0) { return getInstance().mvCamParams[i].p2; }
    static float k3(int i = 0) { return getInstance().mvCamParams[i].k3; }
    static cv::Mat K(int i = 0) { return getInstance().mvK[i]; }
    static cv::Mat distCoeffs(int i = 0) { return getInstance().mvDistCoeffs[i]; }
    ///@}
    /**
     * @name Getters for Feature Extraction Parameters
     * @brief A group of feature extraction parameter getters.
     */
    ///@{
    static int nFeatures() { return getInstance().mFeatParams.nFeatures; }
    static float scaleFactor() { return getInstance().mFeatParams.scaleFactor; }
    static int nLevels() { return getInstance().mFeatParams.nLevels; }
    ///@}
private: // private data
    /// Camera parameters for each camera.
    std::vector<CameraParameters> mvCamParams;
    /// Feature extraction parameters.
    FeatExtParameters mFeatParams;
    /// Camera intrinsics.
    std::vector<cv::Mat> mvK;
    /// Camera distortion coefficients.
    std::vector<cv::Mat> mvDistCoeffs;
    /// Number of camera distorion coefficients.
    std::vector<int> mvnDistCoeffs;
private: // private member functions
    /// Constructor that configures all the parsed system parameters.
    Config() = default;
    /// No copy is allowed.
    Config(const Config&) = delete; 
    /// No copy-assignment is allowed.
    Config& operator=(const Config&) = delete;
    /// Set camera parameters based on loaded cfg file and system mode.
    void setCamParams(const cv::FileStorage& fs, System::Mode eMode);
    /// Set feature extraction parameters based on loaded cfg file.
    void setFeatExtParams(const cv::FileStorage& fs);
    /** 
     * @name Setters for Camera Intrinsics
     * @brief Set camera intrinsics and distortion coefficients 
     *        using configured camera parameters.
     * @param[in] view Camera index starting from 0.
     */
    ///@{
    void setCamIntrinsics(int view);
    void setCamDistCoeffs(int view);
    ///@}
};

/// Display configured system parameters.
std::ostream& operator<<(std::ostream& os, const Config& cfg);

} // namespace SLAM_demo

#endif // CONFIG_HPP
