/**
 * @file   CamDataLoader.hpp
 * @brief  Header of camera data loader class.
 * @author Charlie Li
 * @date   2019.08.05
 */

#ifndef CAM_DATA_LOADER_HPP
#define CAM_DATA_LOADER_HPP

#include <opencv2/core.hpp>

#include <vector>
#include <string>

/**
 * @class CamDataLoader
 * @brief Data loader class for camera image data and the corresponding 
 *        timestamp data.
 */
class CamDataLoader {
public:
    /** 
     * @brief Constructor for monocular case. 
     *
     * The constructor will read the dataset csv data file and
     * get timestamp and image filename info.
     *
     * The csv file should be in the following format:
     * - 2 Columns;
     * - Column name for the 1st row;
     * - Timestamp info in the 1st column and image filename in the 2nd column.
     *
     * @param[in] strImgPath  Input path of images to be loaded.
     * @param[in] strDataFile Input path of the csv data file containing
     *                        timestamp and image filename info.
     */
    CamDataLoader(const std::string& strImgPath,
                  const std::string& strDataFile);
    /// Different views for monocular/stereo/RGB-D cases.
    enum class View {
        MONO,  ///< The single view for monocular case.
        LEFT,  ///< The left view for stereo case.
        RIGHT, ///< The right view for stereo case.
        RGB,   ///< The RGB view for RGB-D case.
        DEPTH  ///< The depth (D) view for RGB-D case.
    };
    /** 
     * @brief Load image data (1 view only) for monocular/stereo/RGB-D cases.
     * @param[in] nFrame Frame index starting from 0.
     * @param[in] eView  The image view to be loaded.
     * @return Loaded image data of cv::Mat type.
     */
    cv::Mat loadImg(int nFrame, View eView) const;
    /// Get number of frames loaded.
    int getNFrames() const { return mvTimestamps.size(); }
    /// Get timestamp info of a target frame.
    double getTimestamp(int nFrame) const { return mvTimestamps[nFrame]; }
private:
    std::vector<std::string> mvstrImgs1; // image filenames (mono/left/RGB view)
    std::vector<std::string> mvstrImgs2; // image filenames (right/D view)
    std::vector<double> mvTimestamps; // timestamp info (unit: second)
};

#endif // CAM_DATA_LOADER_HPP
