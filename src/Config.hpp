/**
 * @file   Config.hpp
 * @brief  Header of SLAM system configuration class.
 * @author Charlie Li
 * @date   2019.08.07
 */

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <memory> // std::unique_ptr
#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include "System.hpp"

namespace SLAM_demo {

/// Camera parameters
struct CameraParameters {
    int w;      ///< Width of captured image.
    int h;      ///< Height of captured image.
    double fx;  ///< Focal length (x-axis).
    double fy;  ///< Focal length (y-axis).
    double cx;  ///< Camera center \f$x_c\f$ in pixel.
    double cy;  ///< Camera center \f$y_c\f$ in pixel.
    double k1;  ///< Camera distortion coefficient \f$k_1\f$.
    double k2;  ///< Camera distortion coefficient \f$k_2\f$.
    double p1;  ///< Camera distortion coefficient \f$p_1\f$.
    double p2;  ///< Camera distortion coefficient \f$p_2\f$.
    double fps; ///< Camera framerate.
};

/**
 * @class Config
 * @brief The class stores SLAM system configurations using singleton 
 *        design pattern.
 */
class Config {
public:
    // friends
    friend std::ostream& operator<<(std::ostream& os, const Config& cfg);
    /// Get the only instance of the class.
    static Config& getInstance()
    {
        static std::unique_ptr<Config> instance(new Config);
        return *instance; // dereferencing to get the reference
    }
    
    // Setters
    /**
     * @brief Set system parameters from a config file.
     *
     * The function will use cv::FileStorage class 
     * for parameter parsing on a YAML file.
     *
     * @param[in] strCfgFile Filename of the configuration file.
     * @param[in] eMode      System mode (enum type).
     * @return A boolean value indicating if the configuration is successful.
     */
    bool setParameters(const std::string& strCfgFile,
                       System::Mode eMode = System::Mode::MONOCULAR);
    
    // getters
    /** 
     * @name groupCamParamGetters
     * @brief A group of camera parameter getters.
     * @param[in] i Get the parameter from the \f$i\f$th camera.
     */
    ///@{ // class member group for doxygen
    static int width(int i = 0) { return getInstance().mvCamParams[i].w; }
    static int height(int i = 0) { return getInstance().mvCamParams[i].h; }
    static double fx(int i = 0) { return getInstance().mvCamParams[i].fx; }
    static double fy(int i = 0) { return getInstance().mvCamParams[i].fy; }
    static double cx(int i = 0) { return getInstance().mvCamParams[i].cx; }
    static double cy(int i = 0) { return getInstance().mvCamParams[i].cy; }
    static double k1(int i = 0) { return getInstance().mvCamParams[i].k1; }
    static double k2(int i = 0) { return getInstance().mvCamParams[i].k2; }
    static double p1(int i = 0) { return getInstance().mvCamParams[i].p1; }
    static double p2(int i = 0) { return getInstance().mvCamParams[i].p2; }
    static double fps(int i = 0) { return getInstance().mvCamParams[i].fps; }
    ///@} // end of groupCamParamGetters
private:
    /// Constructor that configures all the parsed system parameters.
    Config(){}
    /// No copy is allowed.
    Config(const Config&) = delete; 
    /// No copy-assignment is allowed.
    Config& operator=(const Config&) = delete;
    /// Set camera parameters based on loaded cfg file and system mode.
    void setCamParams(const cv::FileStorage& fs, System::Mode eMode);
private: // private data for this class
    /// Camera parameters for each camera.
    std::vector<CameraParameters> mvCamParams; 
};

/// Display configured system parameters.
std::ostream& operator<<(std::ostream& os, const Config& cfg);


} // namespace SLAM_demo

#endif // CONFIG_HPP
