/**
 * @file   LocalMapper.cpp
 * @brief  Implementation of local mapper class in SLAM system.
 * @author Charlie Li
 * @date   2019.10.22
 */

#include "LocalMapper.hpp"

#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "Config.hpp"
#include "FeatureMatcher.hpp"
#include "KeyFrame.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"
#include "Optimizer.hpp"
#include "Utility.hpp"

namespace SLAM_demo {

using cv::Mat;
using std::list;
using std::make_shared;
using std::set;
using std::shared_ptr;
using std::vector;

const int LocalMapper::NUM_BEST_KF = 20;

LocalMapper::LocalMapper(const std::shared_ptr<Map>& pMap,
                         const std::shared_ptr<Optimizer>& pOptimizer) :
    mpMap(pMap), mpOpt(pOptimizer), mpKFCur(nullptr) {}

void LocalMapper::run()
{
    // currently traverse all new keyframes in the list
    while (!mlNewKFs.empty()) {
        // access the keyframes from the front of the list
        mpKFCur = mlNewKFs.front();
        mlNewKFs.pop_front();
        // * triangulation:
        // for each connected KF:
        //   matches2Dto2D <- match(2Dto2D, KF, KFcur)
        //   {Xw} <- triangulate(matches2Dto2D)
        //   fuse(KFcur, KF, matches2Dto2D, {Xw})
        createNewMapPoints();
        // * map point fusion:
        // for each connected KF:
        //   fuse(KFcur, KF) // from KFcur to KF
        // for each connected KF:
        //   fuse(KF, KFcur)
        fuseNewMapPoints();
        // perform local BA
        mpOpt->localBundleAdjustment(mpKFCur, 20, true);
        // remove redundant keyframes
        removeKeyFrames();
    }
    // remove redundant map points
    mpMap->removeMPts();
}

void LocalMapper::insertKeyFrame(const std::shared_ptr<KeyFrame>& pKF)
{
    mlNewKFs.push_back(pKF);
}

void LocalMapper::createNewMapPoints() const
{
    set<shared_ptr<KeyFrame>> spKFs;
    // get best connected keyframes
    vector<shared_ptr<KeyFrame>> vpBestConnectedKFs =
        mpKFCur->getBestConnectedKFs(NUM_BEST_KF);
    spKFs.insert(vpBestConnectedKFs.cbegin(), vpBestConnectedKFs.cend());
    for (const auto& pKF : vpBestConnectedKFs) {
        vector<shared_ptr<KeyFrame>> vpExtraKFs =
            pKF->getBestConnectedKFs(5);
        spKFs.insert(vpExtraKFs.cbegin(), vpExtraKFs.cend());
    }
    // set 2D-to-2D feature matcher
    shared_ptr<FeatureMatcher> pFMatcher = make_shared<FeatureMatcher>(
        10000.0f, false); //true, 0.75f);
    // check number of newly created map points
    int nMPts = 0;
    // traverse each connected keyframe
    for (const auto& pKF : spKFs) {
        // get 2D-to-2D matches
        vector<cv::DMatch> vMatches2Dto2D =
            pFMatcher->match2Dto2D(mpKFCur, pKF);
        // triangulate new map points
        vector<Mat> vXws = Utility::triangulate3DPts(mpKFCur, pKF,
                                                     vMatches2Dto2D);
        nMPts += fuseMapPoints(mpKFCur, pKF, vMatches2Dto2D, vXws);
    }
    std::cout << vpBestConnectedKFs.size() << " best KFs; "
              << nMPts << " map points created" << std::endl;
}

void LocalMapper::fuseNewMapPoints() const
{
    // get all related keyframes: currently all strong-connected KFs
    set<shared_ptr<KeyFrame>> spRelatedKFs;
    // all connected keyframes
    vector<shared_ptr<KeyFrame>> vpConnectedKFs = mpKFCur->getConnectedKFs();
    spRelatedKFs.insert(vpConnectedKFs.cbegin(), vpConnectedKFs.cend());
    //vector<shared_ptr<KeyFrame>> vpWeakKFs = mpKFCur->getWeakKFs();
    //spRelatedKFs.insert(vpWeakKFs.cbegin(), vpWeakKFs.cend());
    // best neighbours of each connected keyframes
    for (const auto& pKF : vpConnectedKFs) {
        vector<shared_ptr<KeyFrame>> vpBestKFs = pKF->getBestConnectedKFs(2);
        spRelatedKFs.insert(vpBestKFs.cbegin(), vpBestKFs.cend());
    }
    // fuse map points of newly added keyframe to each of its related keyframes
    for (const auto& pKF : spRelatedKFs) {
        fuseMapPoints(mpKFCur, pKF);
    }
    // fuse map points of each related keyframe to the newly added keyframe
    for (const auto& pKF : spRelatedKFs) {
        fuseMapPoints(pKF, mpKFCur);
    }
}

void LocalMapper::removeKeyFrames() const
{
    // all connected keyframes
    vector<shared_ptr<KeyFrame>> vpConnectedKFs = mpKFCur->getConnectedKFs();
    // traverse each connected KF for KFs to be removed
    for (const auto& pKF : vpConnectedKFs) {
        set<shared_ptr<MapPoint>> spMPts;
        vector<shared_ptr<MapPoint>> vpMPts = pKF->mappoints();
        for (const auto& pMPt : vpMPts) {
            if (pMPt) {
                spMPts.insert(pMPt);
            }
        }
        // check number of observations for each local map point
        int nRedundant = 0;
        for (const auto& pMPt : spMPts) {
            int nObses = pMPt->getNumObservations();
            if (nObses > 2) {
                ++nRedundant;
            }
        }
        float ratioRedundant = static_cast<float>(nRedundant) / spMPts.size();
        if (ratioRedundant > 0.8f) { // remove data of qualified keyframe
            for (const auto& pMPt : spMPts) {
                pMPt->removeObservation(pKF);
            }
        }
    }
}

int LocalMapper::fuseMapPoints(const std::shared_ptr<KeyFrame>& pKF2,
                               const std::shared_ptr<KeyFrame>& pKF1,
                               const std::vector<cv::DMatch>& vMatches21,
                               const std::vector<cv::Mat>& vXws) const
{
    int nMPts = 0;
    // traverse each match with valid triangulated point
    for (unsigned i = 0; i < vMatches21.size(); ++i) {
        const Mat& Xw = vXws[i];
        if (Xw.empty()) { // check whether valid
            continue;
        }
        // check if new point is better than (possibly bound) current map points
        const cv::DMatch& match21 = vMatches21[i];
        // build new map point if new point is better than both pMPt2 & pMPt1
        if (isNewPtBetter(Xw, pKF2, pKF1, match21)) {
            shared_ptr<MapPoint> pMPt = make_shared<MapPoint>(mpMap, Xw);
            int nIdxKpt2 = match21.queryIdx;
            int nIdxKpt1 = match21.trainIdx;
            // remove observation data for old map points
            //shared_ptr<MapPoint> pMPt2 = pKF2->mappoint(nIdxKpt2);
            //shared_ptr<MapPoint> pMPt1 = pKF1->mappoint(nIdxKpt1);
            //if (pMPt2) {
            //    pMPt2->removeObservation(pKF2);
            //}
            //if (pMPt1) {
            //    pMPt1->removeObservation(pKF1);
            //}
            // add observation data for newly added map point
            pMPt->addObservation(pKF1, nIdxKpt1);
            pMPt->addObservation(pKF2, nIdxKpt2);
            pKF1->bindMPt(pMPt, nIdxKpt1);
            pKF2->bindMPt(pMPt, nIdxKpt2);
            pMPt->updateDescriptor();
            mpMap->addMPt(pMPt);
            ++nMPts;
        }
    }
    // update weighted covisibiity graph
    pKF1->updateConnections();
    pKF2->updateConnections();
    return nMPts;
}

void LocalMapper::fuseMapPoints(const std::shared_ptr<KeyFrame>& pKFsrc,
                                const std::shared_ptr<KeyFrame>& pKFdst) const
{
    // fuse(KFsrc, KFdst):
    //   match(2Dto3D, KFdst, KFsrc):
    //     check reproj error based on KF's pose
    //   fuse same MPts (same keypoint index):
    //     MPt_old.removeObsData()
    //     MPt_new.addObsData()
    //   add new MPt obs data
    shared_ptr<FeatureMatcher> pFMatcher = make_shared<FeatureMatcher>(
        4.0f, false); //true, 0.75f);
    // 2: dst (querying); 1: src (training)
    vector<cv::DMatch> vMatches21 = pFMatcher->match2Dto3D(
        pKFdst, pKFsrc, false /* do not update map point bindings */);
    // fuse existed map point observations / add new map point observations
    for (const auto& match21 : vMatches21) {
        int nKptsrc = match21.trainIdx;
        int nKptdst = match21.queryIdx;
        shared_ptr<MapPoint> pMPt = pKFsrc->mappoint(nKptsrc);
        shared_ptr<MapPoint> pMPtOld = pKFdst->mappoint(nKptdst);
        if (!pMPtOld) { // add new observation data
            pMPt->addObservation(pKFdst, nKptdst);
            pKFdst->bindMPt(pMPt, nKptdst);
            pMPt->updateDescriptor();
        } else { // fuse observation data
            // keep map point with more observations
            if (pMPt->getNumObservations() > pMPtOld->getNumObservations()) {
                // fuse new map point
                pMPtOld->removeObservation(pKFdst);
                pMPt->addObservation(pKFdst, nKptdst);
                pKFdst->bindMPt(pMPt, nKptdst);
                pMPtOld->updateDescriptor();
                pMPt->updateDescriptor();
            }
        }
    }
    // update weighted covisibility graph
    pKFsrc->updateConnections();
    pKFdst->updateConnections();
}

bool LocalMapper::isNewPtBetter(const cv::Mat& Xw,
                                const std::shared_ptr<KeyFrame>& pKF2,
                                const std::shared_ptr<KeyFrame>& pKF1,
                                const cv::DMatch& match21) const
{
    int nIdxKpt2 = match21.queryIdx;
    int nIdxKpt1 = match21.trainIdx;
    shared_ptr<MapPoint> pMPt2 = pKF2->mappoint(nIdxKpt2);
    shared_ptr<MapPoint> pMPt1 = pKF1->mappoint(nIdxKpt1);
    bool bMPt2Bad = !pMPt2 || pMPt2->isOutlier();
    bool bMPt1Bad = !pMPt1 || pMPt1->isOutlier();
    if (bMPt2Bad && bMPt1Bad) { // only new point is valid
        return true;
    }
    if (bMPt2Bad) { // check if pMPt1 can be replaced
        
    }
    if (bMPt1Bad) { // check if pMPt2 can be replaced
        
    }
    return true;
}


} // namespace SLAM_demo
