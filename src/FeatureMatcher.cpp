/**
 * @file   FeatureMatcher.cpp
 * @brief  Implementation of feature matcher class in SLAM system.
 * @author Charlie Li
 * @date   2019.10.22
 */

#include "FeatureMatcher.hpp"

#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "Config.hpp"
#include "FrameBase.hpp"
#include "MapPoint.hpp"
#include "Utility.hpp"

namespace SLAM_demo {

using cv::Mat;
using std::make_shared;
using std::map;
using std::shared_ptr;
using std::vector;

FeatureMatcher::FeatureMatcher(float thDistMatchMax, bool bUseLoweRatioTest,
                               float thRatioTest) :
    mThDistMatchMax(thDistMatchMax), mbUseLoweRatioTest(bUseLoweRatioTest),
    mThRatioTest(thRatioTest)
{
    // initialize feature matcher
    if (mbUseLoweRatioTest) {
        // FLANN-based matcher & use Lowe's ratio test
        mpFeatMatcher = make_shared<cv::FlannBasedMatcher>(
            cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    } else {
        // use symmetric test
        mpFeatMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    }
}

std::vector<cv::DMatch> FeatureMatcher::match2Dto2D(
    const std::shared_ptr<FrameBase>& pF2,
    const std::shared_ptr<FrameBase>& pF1) const
{
    unsigned nBestMatches = mbUseLoweRatioTest ? 2 : 1;
    vector<vector<cv::DMatch>> vvMatches21;
    mpFeatMatcher->knnMatch(pF2->descriptors(), // query
                            pF1->descriptors(), // train
                            vvMatches21,
                            nBestMatches); // number of best matches
    return filterMatchResult(vvMatches21, pF2, pF1);
}

std::vector<cv::DMatch> FeatureMatcher::match2Dto3D(
    const std::shared_ptr<FrameBase>& pF2,
    const std::shared_ptr<FrameBase>& pF1,
    bool bBindMPts) const
{
    unsigned nBestMatches = mbUseLoweRatioTest ? 2 : 1;
    // get matching mask
    map<int, shared_ptr<MapPoint>> mpMPts1 = pF1->getMPtsMap();
    //const vector<cv::KeyPoint>& vKpts1 = pF1->keypoints();
    //const vector<cv::KeyPoint>& vKpts2 = pF2->keypoints();
    //vector<int> vIdxKpts1;
    //vIdxKpts1.reserve(mpMPts1.size());
    //for (const auto& pair : mpMPts1) {
    //    vIdxKpts1.push_back(pair.first);
    //}
    //Mat mask = getMatchMask2Dto3D(vIdxKpts1, vKpts2.size(), vKpts1.size());
    vector<bool> vbMPtsValid(pF1->keypoints().size(), false);
    for (const auto& pair : mpMPts1) {
        vbMPtsValid[pair.first] = true;
    }
    // feature matching with mask (partial query set and full train set)
    vector<vector<cv::DMatch>> vvMatches21;
    mpFeatMatcher->knnMatch(pF2->descriptors(), // query
                            pF1->descriptors(), // train
                            vvMatches21,
                            nBestMatches,
                            //mask, true); // mask is not functioning!!!
                            cv::noArray(), false);
    vector<cv::DMatch> vMatches21 = filterMatchResult(vvMatches21, pF2, pF1,
                                                      vbMPtsValid);
    // update map point data on current frame (pF2)
    if (bBindMPts) {
        for (const auto& match21 : vMatches21) {
            shared_ptr<MapPoint> pMPt = pF1->mappoint(match21.trainIdx);
            pF2->bindMPt(pMPt, match21.queryIdx);
        }
    }
    return vMatches21;
}

std::vector<cv::DMatch> FeatureMatcher::match2Dto3D(
    const std::shared_ptr<FrameBase>& pF2,
    const std::vector<std::shared_ptr<MapPoint>>& vpMPts,
    bool bBindMPts) const
{
    unsigned nBestMatches = mbUseLoweRatioTest ? 2 : 1;
    // get all descriptors
    Mat descXws;
    for (const auto& pMPt : vpMPts) {
        descXws.push_back(pMPt->descriptor());
    }
    if (descXws.empty()) {
        return vector<cv::DMatch>();
    }
    // feature matching with mask (partial query set and full train set)
    vector<vector<cv::DMatch>> vvMatches21;
    mpFeatMatcher->knnMatch(pF2->descriptors(), // query
                            descXws, // train
                            vvMatches21,
                            nBestMatches);
    vector<cv::DMatch> vMatches21 = filterMatchResult(vvMatches21, pF2,
                                                      vpMPts);
    // update map point data on current frame (pF2)
    if (bBindMPts) {
        for (const auto& match21 : vMatches21) {
            shared_ptr<MapPoint> pMPt = vpMPts[match21.trainIdx];
            pF2->bindMPt(pMPt, match21.queryIdx);
        }
    }
    return vMatches21;    
}

std::vector<cv::DMatch> FeatureMatcher::filterMatchResult(
    const std::vector<std::vector<cv::DMatch>>& vvMatches21,
    const std::shared_ptr<FrameBase>& pF2,
    const std::shared_ptr<FrameBase>& pF1,
    const std::vector<bool>& vbMask1) const
{
    vector<cv::DMatch> vMatches21;
    vMatches21.reserve(vvMatches21.size());
    unsigned nBestMatches = mbUseLoweRatioTest ? 2 : 1;
    const vector<cv::KeyPoint>& vKpts1 = pF1->keypoints();
    const vector<cv::KeyPoint>& vKpts2 = pF2->keypoints();
    bool b2Dto2DCase = vbMask1.empty() ? true : false;
    for (unsigned i = 0; i < vvMatches21.size(); ++i) {
        // skip invalid matching result
        if (vvMatches21[i].size() != nBestMatches) {
            continue;
        }
        // skip null map point
        if (!b2Dto2DCase && !vbMask1[vvMatches21[i][0].trainIdx]) {
            continue;
        }
        // Lowe's ratio test
        if (nBestMatches == 2 &&
            vvMatches21[i][0].distance >=
            mThRatioTest * vvMatches21[i][1].distance) {
            continue;
        }
        // filter out-of-border matches
        const cv::Point2f& pt1 = vKpts1[vvMatches21[i][0].trainIdx].pt;
        const cv::Point2f& pt2 = vKpts2[vvMatches21[i][0].queryIdx].pt;
        if (!Utility::is2DPtInBorder(Mat(pt1)) &&
            !Utility::is2DPtInBorder(Mat(pt2))) {
            continue;
        }
        // filter matches whose dist between (reprojected) 2D point in view 1
        // and 2D point in view 2 is larger than a threshold
        Mat x1;
        if (b2Dto2DCase) {
            x1 = Mat(pt1);
        } else {
            const shared_ptr<MapPoint> pMPt =
                pF1->mappoint(vvMatches21[i][0].trainIdx);
            assert(pMPt);
            x1 = pF1->coordWorld2Img(pMPt->X3D());
        }
        Mat x2 = Mat(pt2);
        Mat xDiff = x1 - x2;
        float xDistSq = xDiff.dot(xDiff);
        if (xDistSq > mThDistMatchMax * mThDistMatchMax) {
            continue;
        }
        vMatches21.push_back(vvMatches21[i][0]);
    }    
    return vMatches21;
}

std::vector<cv::DMatch> FeatureMatcher::filterMatchResult(
    const std::vector<std::vector<cv::DMatch>>& vvMatches21,
    const std::shared_ptr<FrameBase>& pF2,
    const std::vector<std::shared_ptr<MapPoint>>& vpMPts) const
{
    vector<cv::DMatch> vMatches21;
    vMatches21.reserve(vvMatches21.size());
    unsigned nBestMatches = mbUseLoweRatioTest ? 2 : 1;
    const vector<cv::KeyPoint>& vKpts2 = pF2->keypoints();
    for (unsigned i = 0; i < vvMatches21.size(); ++i) {
        // skip invalid matching result
        if (vvMatches21[i].size() != nBestMatches) {
            continue;
        }
        // Lowe's ratio test
        if (nBestMatches == 2 &&
            vvMatches21[i][0].distance >=
            mThRatioTest * vvMatches21[i][1].distance) {
            continue;
        }
        // filter out-of-border matches
        const shared_ptr<MapPoint>& pMPt = vpMPts[vvMatches21[i][0].trainIdx];
        Mat x1 = pF2->coordWorld2Img(pMPt->X3D());
        const cv::Point2f& pt1 = cv::Point2f(x1.at<float>(0), x1.at<float>(1));
        const cv::Point2f& pt2 = vKpts2[vvMatches21[i][0].queryIdx].pt;
        if (!Utility::is2DPtInBorder(Mat(pt1)) &&
            !Utility::is2DPtInBorder(Mat(pt2))) {
            continue;
        }
        // filter matches whose dist between (reprojected) 2D point in view 1
        // and 2D point in view 2 is larger than a threshold
        Mat x2 = Mat(pt2);
        Mat xDiff = x1 - x2;
        float xDistSq = xDiff.dot(xDiff);
        if (xDistSq > mThDistMatchMax * mThDistMatchMax) {
            continue;
        }
        vMatches21.push_back(vvMatches21[i][0]);
    }    
    return vMatches21;
}

cv::Mat FeatureMatcher::getMatchMask2Dto3D(const std::vector<int>& vIdxKpts1,
                                           int nKpts2, int nKpts1) const
{
    Mat mask; // {row = vIdxKpts1.size(), col = nNumKpts2, type = CV_8UC1}
    int idx = 0;
    int nMPts = vIdxKpts1.size();
    for (int i = 0; i < nKpts1; ++i) {
        bool bAllowMatching = false;
        if (idx == nMPts) { // last kpt index traversed
            bAllowMatching = false;
        } else {
            if (i < vIdxKpts1[idx]) { // less than nearest kpt index
                bAllowMatching = false;
            } else { // kpt index with map point bound
                bAllowMatching = true;
            }
        }
        // add one row to the transpose of mask
        if (bAllowMatching) {
            mask.push_back(Mat::ones(1, nKpts2, CV_8UC1));
            ++idx;
        } else {
            mask.push_back(Mat::zeros(1, nKpts2, CV_8UC1));
        }
    }
    mask = mask.t();
    assert(mask.rows == nKpts2 && mask.cols == nKpts1);
    return mask;
}

} // namespace SLAM_demo
