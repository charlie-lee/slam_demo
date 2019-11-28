/**
 * @file   Optimizer.cpp
 * @brief  Implementations of Optimizer class for pose & map data optimization.
 * @author Charlie Li
 * @date   2019.09.17
 */

#include "Optimizer.hpp"

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
//#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h> // g2o::RobustKernelHuber, ...
#include <g2o/core/solver.h>
//#include <g2o/solvers/cholmod/linear_solver_cholmod.h> // for global BA
#include <g2o/solvers/csparse/linear_solver_csparse.h> // for global BA
#include <g2o/solvers/dense/linear_solver_dense.h> // for pose optimization
#include <g2o/solvers/eigen/linear_solver_eigen.h> // for local BA
#include <g2o/types/sba/types_six_dof_expmap.h> // g2o::EdgeSE3ProjectXYZ, ...
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp> // cv::cv2eigen()
#include <Eigen/Core>
#include <Eigen/Geometry> // Eigen::Quaternion
#include "Config.hpp"
#include "Frame.hpp"
#include "KeyFrame.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"
#include "Tracker.hpp"

namespace SLAM_demo {

using std::map;
using std::make_shared;
using std::set;
using std::shared_ptr;
using std::vector;
using cv::Mat;

const int Optimizer::TH_MIN_NUM_MAPPOINT = 5;
const float Optimizer::TH_MAX_CHI2_FACTOR = 9.0f;

Optimizer::Optimizer(const std::shared_ptr<Map>& pMap) : mpMap(pMap) {}

int Optimizer::globalBundleAdjustment(unsigned nKFs, int nIter,
                                      bool bRobust) const
{
    // set definition of solver
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver;
    pLinearSolver =
        new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* pBlkSolver;
    pBlkSolver = new g2o::BlockSolver_6_3(pLinearSolver);
    g2o::OptimizationAlgorithmLevenberg* pSolverLM;
    pSolverLM = new g2o::OptimizationAlgorithmLevenberg(pBlkSolver);
    // configure optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(pSolverLM);
    
    // add vertices: poses
    vector<shared_ptr<KeyFrame>> vpKFs = mpMap->getLastNKFs(nKFs);
    unsigned idxKFMax = 0;
    for (const auto& pKF : vpKFs) {
        g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
        Mat Tcw = pKF->mPose.getPose();
        unsigned idxKF = pKF->index();
        pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        pVSE3->setId(idxKF);
        bool bFixed = idxKF == 0;
        pVSE3->setFixed(bFixed);
        optimizer.addVertex(pVSE3);
        if (idxKF > idxKFMax) {
            idxKFMax = idxKF;
        }
    }
    // add vertices: map points, and add edges for each map point
    vector<shared_ptr<MapPoint>> vpMPts = mpMap->getAllMPts();
    int nMPts = vpMPts.size();
    // skip optimization if map points are not enough
    if (nMPts < TH_MIN_NUM_MAPPOINT) {
        return 0;
    }
    vector<bool> vbMPtOptimized(nMPts, true);
    vector<vector<g2o::EdgeSE3ProjectXYZ*>> vvpEdges(nMPts);    
    for (int i = 0; i < nMPts; ++i) {
        const shared_ptr<MapPoint>& pMPt = vpMPts[i];
        g2o::VertexSBAPointXYZ* pVPt = new g2o::VertexSBAPointXYZ();
        pVPt->setEstimate(cvMat2Vector3d(pMPt->X3D()));
        pVPt->setId(idxKFMax + 1 + i);
        pVPt->setMarginalized(true); // why?? (to decrease the size of Hessian?)
        optimizer.addVertex(pVPt);
        
        // add edges
        vector<shared_ptr<KeyFrame>> vpKFsMPt = (nKFs == 0) ?
            pMPt->getRelatedKFs() : vpKFs;
        int nKFsMpt = vpKFsMPt.size();
        vvpEdges[i].resize(nKFsMpt, nullptr);
        bool bHasEdge = false; // check whether the vertec has edges
        for (int j = 0; j < nKFsMpt; ++j) {
            auto& pKF = vpKFsMPt[j];        
            // check whether the map point is observed by the target keyframe
            if (!pMPt->isObservedBy(pKF)) {
                continue;
            }
            // form an edge
            bHasEdge = true;
            cv::KeyPoint kpt = pMPt->keypoint(pKF);
            Eigen::Matrix<double, 2, 1> obs;
            obs << kpt.pt.x, kpt.pt.y;
            g2o::EdgeSE3ProjectXYZ* pEdge = new g2o::EdgeSE3ProjectXYZ();
            // vertex 0: map point
            pEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 pVPt));
            // vertex 1: pose
            pEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 optimizer.vertex(pKF->index())));
            pEdge->setMeasurement(obs);
            // set element in information matrix (value = 1 / sigma^2)
            float sigma = std::pow(Config::scaleFactor(), kpt.octave);
            float invSigma2 = 1.0f / (sigma * sigma);
            pEdge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
            // set robust kernel
            if (bRobust) {
                g2o::RobustKernelHuber* pRK = new g2o::RobustKernelHuber();
                pEdge->setRobustKernel(pRK);
                // rho(x) = x^2 if |x| < delta else 2*delta*|x| - delta^2
                pRK->setDelta(sigma);
            }
            // set cam intrinsics
            pEdge->fx = Config::fx();
            pEdge->fy = Config::fy();
            pEdge->cx = Config::cx();
            pEdge->cy = Config::cy();
            // record all the edges
            vvpEdges[i][j] = pEdge;
            // only optimize outliers at the last iteration
            if (pMPt->isOutlier()) {
                pEdge->setLevel(1);
            }            
            optimizer.addEdge(pEdge);
        }
        if (!bHasEdge) {
            optimizer.removeVertex(pVPt);
            vbMPtOptimized[i] = false;
        }
    }

    // optimize
    int nIt = 2;
    for (int it = 0; it < nIt; ++it) {
        optimizer.initializeOptimization(0);
        optimizer.optimize(nIter);
        // exclude outliers
        for (int i = 0; i < nMPts; ++i) {
            auto& pMPt = vpMPts[i];
            if (vbMPtOptimized[i]) {
                vector<shared_ptr<KeyFrame>> vpKFsMPt = (nKFs == 0) ?
                    pMPt->getRelatedKFs() : vpKFs;
                int nKFsMpt = vpKFsMPt.size();
                for (int j = 0; j < nKFsMpt; ++j) {
                    auto& pEdge = vvpEdges[i][j];
                    if (!pEdge) {
                        continue;
                    }
                    // optimize all edges for the last iteration
                    // exclude outliers for other iterations
                    if (it == nIt - 2) {
                        pEdge->setLevel(0);
                    } else {
                        float chi2 = pEdge->chi2();
                        cv::KeyPoint kpt = pMPt->keypoint(vpKFsMPt[j]);
                        float sigma = std::pow(Config::scaleFactor(),
                                               kpt.octave);
                        if (chi2 > sigma*sigma ||
                            !pEdge->isDepthPositive()) {
                            pEdge->setLevel(1);
                        } else {
                            pEdge->setLevel(0);
                        }
                    }
                }
            }
        }
    }
    
    // update results back to the keyframes/map
    // pose update via vpKFs
    for (const auto& pKF : vpKFs) {
        g2o::VertexSE3Expmap* pVSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(pKF->index()));
        g2o::SE3Quat T = pVSE3->estimate();
        pKF->mPose.setPose(SE3Quat2cvMat(T));
    }
    // map point data update via vpMPts
    // count the number of map point outliers after optimization
    int nInliers = nMPts;
    for (int i = 0; i < nMPts; ++i) {
        if (vbMPtOptimized[i]) {
            shared_ptr<MapPoint>& pMPt = vpMPts[i];
            g2o::VertexSBAPointXYZ* pVPt = dynamic_cast<
                g2o::VertexSBAPointXYZ*>(optimizer.vertex(idxKFMax + 1 + i));
            Eigen::Vector3d X = pVPt->estimate();
            pMPt->setX3D(Vector3d2cvMat(X));
            
            // set outlier status for all map points
            pMPt->setOutlier(false);
            vector<shared_ptr<KeyFrame>> vpKFsMPt = (nKFs == 0) ?
                pMPt->getRelatedKFs() : vpKFs;
            int nKFsMpt = vpKFsMPt.size();
            for (int j = 0; j < nKFsMpt; ++j) {
                auto& pEdge = vvpEdges[i][j];
                if (!pEdge) {
                    continue;
                }
                float chi2 = pEdge->chi2();
                cv::KeyPoint kpt = pMPt->keypoint(vpKFsMPt[j]);
                float sigma = std::pow(Config::scaleFactor(), kpt.octave);
                if (chi2 > sigma*sigma * TH_MAX_CHI2_FACTOR ||
                    !pEdge->isDepthPositive()) {
                    pMPt->setOutlier(true);
                    --nInliers;
                    break;
                }
            }            
        }
    }

    // clean up resources
    optimizer.clear();
    optimizer.clearParameters();

    return nInliers;
}

//int Optimizer::frameBundleAdjustment(unsigned nFrames, int nIter,
//                                     bool bRobust) const
//{
//    // set definition of solver
//    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver;
//    pLinearSolver =
//        new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
//    g2o::BlockSolver_6_3* pBlkSolver;
//    pBlkSolver = new g2o::BlockSolver_6_3(pLinearSolver);
//    g2o::OptimizationAlgorithmLevenberg* pSolverLM;
//    pSolverLM = new g2o::OptimizationAlgorithmLevenberg(pBlkSolver);
//    // configure optimizer
//    g2o::SparseOptimizer optimizer;
//    optimizer.setVerbose(false);
//    optimizer.setAlgorithm(pSolverLM);
//
//    // add vertices: poses
//    vector<shared_ptr<Frame>> vpKFs = mpMap->getLastNKFs(nFrames);
//    shared_ptr<Frame> pFrameCur = nullptr;
//
//    unsigned idxFMax = 0;
//    for (const auto& pFrame : vpFrames) {
//        g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
//        Mat Tcw = pFrame->mPose.getPose();
//        unsigned idxF = pFrame->index();
//        pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
//        pVSE3->setId(idxF);
//        bool bFixed = idxF != System::nCurrentFrame;
//        if (!bFixed) {
//            pFrameCur = pFrame;
//        }
//        pVSE3->setFixed(bFixed);
//        optimizer.addVertex(pVSE3);
//        if (idxF > idxFMax) {
//            idxFMax = idxF;
//        }
//    }
//
//    // no target frame to be optimized
//    if (!pFrameCur) {
//        return 0;
//    }
//
//    // add vertices: map points, and add edges for each map point
//    vector<shared_ptr<MapPoint>> vpMPts = pFrameCur->getpMPtsObserved();
//    int nMPts = vpMPts.size();
//    // skip optimization if map points are not enough
//    if (nMPts < TH_MIN_NUM_MAPPOINT) {
//        return 0;
//    }    
//    vector<vector<g2o::EdgeSE3ProjectXYZ*>> vvpEdges(nMPts);    
//    for (int i = 0; i < nMPts; ++i) {
//        const shared_ptr<MapPoint>& pMPt = vpMPts[i];
//        g2o::VertexSBAPointXYZ* pVPt = new g2o::VertexSBAPointXYZ();
//        pVPt->setEstimate(cvMat2Vector3d(pMPt->X3D()));
//        pVPt->setId(idxFMax + 1 + i);
//        pVPt->setMarginalized(true);
//        optimizer.addVertex(pVPt);
//        
//        // add edges
//        vvpEdges[i].resize(nFrames, nullptr);
//        for (unsigned j = 0; j < nFrames; ++j) {
//            auto& pFrame = vpFrames[j];
//            // check whether the map point is observed by the target frame
//            if (!pMPt->isObservedBy(pFrame)) {
//                continue;
//            }
//            cv::KeyPoint kpt = pMPt->keypoint(pFrame);
//            Eigen::Matrix<double, 2, 1> obs;
//            obs << kpt.pt.x, kpt.pt.y;
//            g2o::EdgeSE3ProjectXYZ* pEdge = new g2o::EdgeSE3ProjectXYZ();
//            // vertex 0: map point
//            pEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
//                                 pVPt));
//            // vertex 1: pose
//            pEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
//                                 optimizer.vertex(pFrame->index())));
//            pEdge->setMeasurement(obs);
//            // set element in information matrix (value = 1 / sigma^2)
//            float sigma = std::pow(Config::scaleFactor(), kpt.octave);
//            float invSigma2 = 1.0f / (sigma * sigma);
//            pEdge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
//            // set robust kernel
//            if (bRobust) {
//                g2o::RobustKernelHuber* pRK = new g2o::RobustKernelHuber();
//                pEdge->setRobustKernel(pRK);
//                // rho(x) = x^2 if |x| < delta else 2*delta*|x| - delta^2
//                pRK->setDelta(sigma);
//            }
//            // set cam intrinsics
//            pEdge->fx = Config::fx();
//            pEdge->fy = Config::fy();
//            pEdge->cx = Config::cx();
//            pEdge->cy = Config::cy();
//            // only optimize outliers at the last iteration
//            if (pMPt->isOutlier()) {
//                pEdge->setLevel(1);
//            }
//            // record all the edges
//            vvpEdges[i][j] = pEdge;
//            optimizer.addEdge(pEdge);
//        }
//    }
//
//    // optimize
//    //optimizer.initializeOptimization();
//    //optimizer.optimize(nIter);
//    int nIt = 3;
//    for (int it = 0; it < nIt; ++it) {
//        optimizer.initializeOptimization(0);
//        optimizer.optimize(nIter);
//        // exclude outliers
//        for (int i = 0; i < nMPts; ++i) {
//            auto& pMPt = vpMPts[i];
//            for (unsigned j = 0; j < nFrames; ++j) {
//                auto& pEdge = vvpEdges[i][j];
//                if (!pEdge) {
//                    continue;
//                }
//                // optimize all edges for the last iteration
//                // exclude outliers for other iterations
//                if (it == nIt - 2) {
//                    pEdge->setLevel(0);
//                } else {
//                    float chi2 = pEdge->chi2();
//                    cv::KeyPoint kpt = pMPt->keypoint(vpFrames[j]);
//                    float sigma = std::pow(Config::scaleFactor(),
//                                           kpt.octave);
//                    if (chi2 > sigma*sigma ||
//                        !pEdge->isDepthPositive()) {
//                        pEdge->setLevel(1);
//                    } else {
//                        pEdge->setLevel(0);
//                    }
//                }
//            }
//        }
//    }
//    
//    // update results back to the frames/map
//    // pose update via vpFrames
//    g2o::VertexSE3Expmap* pVSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(
//        optimizer.vertex(System::nCurrentFrame));
//    g2o::SE3Quat T = pVSE3->estimate();
//    pFrameCur->mPose.setPose(SE3Quat2cvMat(T));
//    // map point data update via vpMPts
//    // count the number of map point outliers after optimization
//    int nInliers = nMPts;
//    for (int i = 0; i < nMPts; ++i) {
//        shared_ptr<MapPoint>& pMPt = vpMPts[i];
//        g2o::VertexSBAPointXYZ* pVPt = dynamic_cast<g2o::VertexSBAPointXYZ*>(
//            optimizer.vertex(idxFMax + 1 + i));
//        Eigen::Vector3d X = pVPt->estimate();
//        pMPt->setX3D(Vector3d2cvMat(X));
//        
//        // set outlier status for all map points
//        pMPt->setOutlier(false);
//        for (unsigned j = 0; j < nFrames; ++j) {
//            auto& pEdge = vvpEdges[i][j];
//            if (!pEdge) {
//                continue;
//            }
//            float chi2 = pEdge->chi2();
//            cv::KeyPoint kpt = pMPt->keypoint(vpFrames[j]);
//            float sigma = std::pow(Config::scaleFactor(), kpt.octave);
//            if (chi2 > sigma*sigma * TH_MAX_CHI2_FACTOR ||
//                !pEdge->isDepthPositive()) {
//                pMPt->setOutlier(true);
//                --nInliers;
//                break;
//            }
//        }
//    }
//
//    // clean up resources
//    optimizer.clear();
//    optimizer.clearParameters();
//
//    return nInliers;
//}

int Optimizer::poseOptimization(const std::shared_ptr<Frame>& pFrame) const
{
    // count the number of map point outliers after optimization
    int nInliers = 0;
    // set definition of solver
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver;
    pLinearSolver =
        new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* pBlkSolver;
    pBlkSolver = new g2o::BlockSolver_6_3(pLinearSolver);
    g2o::OptimizationAlgorithmLevenberg* pSolverLM;
    pSolverLM = new g2o::OptimizationAlgorithmLevenberg(pBlkSolver);
    // configure optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(pSolverLM);
    
    // add vertices: the pose to be optimized
    g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
    Mat Tcw = pFrame->mPose.getPose();
    pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
    pVSE3->setId(0);
    pVSE3->setFixed(false);
    optimizer.addVertex(pVSE3);
    
    // add edges: unary edge with map point data as measurement
    map<int, shared_ptr<MapPoint>> mpMPts = pFrame->getMPtsMap();
    int nMPts = mpMPts.size();
    // skip optimization if map points are not enough
    if (nMPts < TH_MIN_NUM_MAPPOINT) {
        return 0;
    }
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdges;
    vpEdges.reserve(nMPts);
    for (const auto& pair : mpMPts) {
        const shared_ptr<MapPoint>& pMPt = pair.second;
        // is it necessary to check whether the frame is valid?
        cv::KeyPoint kpt = pFrame->keypoint(pair.first);
        Eigen::Matrix<double, 2, 1> obs;
        obs << kpt.pt.x, kpt.pt.y;
        g2o::EdgeSE3ProjectXYZOnlyPose* pEdge =
            new g2o::EdgeSE3ProjectXYZOnlyPose();
        // vertex 0: pose
        pEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                             pVSE3));
        pEdge->setMeasurement(obs);
        // set element in information matrix (value = 1 / sigma^2)
        float sigma = std::pow(Config::scaleFactor(), kpt.octave);
        float invSigma2 = 1.0f / (sigma * sigma);
        pEdge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
        // set robust kernel
        g2o::RobustKernelHuber* pRK = new g2o::RobustKernelHuber();
        pEdge->setRobustKernel(pRK);
        // rho(x) = x^2 if |x| < delta else 2*delta*|x| - delta^2
        pRK->setDelta(sigma);
        // set map point position
        pEdge->Xw = cvMat2Vector3d(pMPt->X3D());
        // set cam intrinsics
        pEdge->fx = Config::fx();
        pEdge->fy = Config::fy();
        pEdge->cx = Config::cx();
        pEdge->cy = Config::cy();
        vpEdges.push_back(pEdge);
        // only optimize outliers at the last iteration
        //if (pMPt->isOutlier()) {
        //    pEdge->setLevel(1);
        //}
        optimizer.addEdge(pEdge);
    }

    // optimize (multi-pass?)
    int nIt = 4;
    for (int it = 0; it < nIt; ++it) {
        //pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        optimizer.initializeOptimization(0); // only optimize level 0 edges
        optimizer.optimize(10);
        // exclude outliers
        auto cit = mpMPts.cbegin();
        for (int i = 0; i < nMPts; ++i, ++cit) {
            assert(cit != mpMPts.cend());
            auto& pEdge = vpEdges[i];
            // optimize all edges for the last iteration
            // exclude outliers for other iterations
            if (it == nIt - 2) {
                pEdge->setLevel(0);
            } else {
                float chi2 = pEdge->chi2();
                cv::KeyPoint kpt = pFrame->keypoint(cit->first);
                float sigma = std::pow(Config::scaleFactor(), kpt.octave);
                if (chi2 > sigma*sigma || !pEdge->isDepthPositive()) {
                    pEdge->setLevel(1);
                } else {
                    pEdge->setLevel(0);
                }
            }
        }
    }

    // update optimized pose and map point outlier status
    auto cit = mpMPts.cbegin();
    for (int i = 0; i < nMPts ; ++i, ++cit) {
        assert(cit != mpMPts.cend());
        auto& pMPt = cit->second;
        auto& pEdge = vpEdges[i];
        float chi2 = pEdge->chi2();
        cv::KeyPoint kpt = pFrame->keypoint(cit->first);
        float sigma = std::pow(Config::scaleFactor(), kpt.octave);
        if (chi2 > sigma*sigma * TH_MAX_CHI2_FACTOR ||
            !pEdge->isDepthPositive()) {
            pMPt->setOutlier(true);
        } else {
            pMPt->setOutlier(false);
            nInliers++;
        }
    }
    g2o::SE3Quat T = pVSE3->estimate();
    pFrame->mPose.setPose(SE3Quat2cvMat(T));

    // clean up resources
    optimizer.clear();
    optimizer.clearParameters();

    return nInliers;
}

void Optimizer::localBundleAdjustment(const std::shared_ptr<KeyFrame>& pKFin,
                                      int nIter, bool bRobust) const
{
    // set definition of solver
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver;
    pLinearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* pBlkSolver;
    pBlkSolver = new g2o::BlockSolver_6_3(pLinearSolver);
    g2o::OptimizationAlgorithmLevenberg* pSolverLM;
    pSolverLM = new g2o::OptimizationAlgorithmLevenberg(pBlkSolver);
    // configure optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(pSolverLM);
    
    // add vertices: poses that can be optimized
    vector<shared_ptr<KeyFrame>> vpConnectedKFs = pKFin->getConnectedKFs();
    set<shared_ptr<KeyFrame>> spKFs;
    spKFs.insert(pKFin);
    spKFs.insert(vpConnectedKFs.cbegin(), vpConnectedKFs.cend());
    set<shared_ptr<MapPoint>> spMPts; // temp container for all local map points
    unsigned idxKFMax = 0;
    for (const auto& pKF : spKFs) {
        g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
        Mat Tcw = pKF->mPose.getPose();
        unsigned idxKF = pKF->index();
        pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        pVSE3->setId(idxKF);
        bool bFixed = pKF->index() == 0;
        pVSE3->setFixed(bFixed);
        optimizer.addVertex(pVSE3);
        if (idxKF > idxKFMax) {
            idxKFMax = idxKF;
        }
        // get local map points
        vector<shared_ptr<MapPoint>> vpMPtsKF = pKF->mappoints();
        for (const auto& pMPt : vpMPtsKF) {
            if (pMPt) {
                spMPts.insert(pMPt);
            }
        }
    }
    // add vertices: fixed poses
    vector<shared_ptr<KeyFrame>> vpWeakKFs = pKFin->getWeakKFs();
    set<shared_ptr<KeyFrame>> spFixedKFs;
    spFixedKFs.insert(vpWeakKFs.cbegin(), vpWeakKFs.cend());
    for (const auto& pMPt: spMPts) {
        vector<shared_ptr<KeyFrame>> vpKFRelated = pMPt->getRelatedKFs();
        for (const auto& pKF : vpKFRelated) {
            // add poses that is not in the local map to the fixed pose set
            if (spKFs.find(pKF) == spKFs.end()) {
                spFixedKFs.insert(pKF);
            }
        }
    }
    for (const auto& pKF : spFixedKFs) {
        g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
        Mat Tcw = pKF->mPose.getPose();
        unsigned idxKF = pKF->index();
        pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        pVSE3->setId(idxKF);
        pVSE3->setFixed(true);
        optimizer.addVertex(pVSE3);
        if (idxKF > idxKFMax) {
            idxKFMax = idxKF;
        }
    }

    // add vertices: map points, and add edges for each map point
    int nMPts = spMPts.size();
    // skip optimization if map points are not enough
    if (nMPts < TH_MIN_NUM_MAPPOINT) {
        return;
    }
    // use std::vector for indexing on each map point
    vector<shared_ptr<MapPoint>> vpMPts(spMPts.cbegin(), spMPts.cend());
    vector<bool> vbMPtOptimized(nMPts, true);
    vector<vector<g2o::EdgeSE3ProjectXYZ*>> vvpEdges(nMPts);    
    for (int i = 0; i < nMPts; ++i) {
        const shared_ptr<MapPoint>& pMPt = vpMPts[i];
        g2o::VertexSBAPointXYZ* pVPt = new g2o::VertexSBAPointXYZ();
        pVPt->setEstimate(cvMat2Vector3d(pMPt->X3D()));
        pVPt->setId(idxKFMax + 1 + i);
        pVPt->setMarginalized(true); // why?? (to decrease the size of Hessian?)
        optimizer.addVertex(pVPt);
        
        // add edges
        vector<shared_ptr<KeyFrame>> vpKFsMPt = pMPt->getRelatedKFs();
        int nKFsMpt = vpKFsMPt.size();
        vvpEdges[i].resize(nKFsMpt, nullptr);
        bool bHasEdge = false; // check whether the vertec has edges
        for (int j = 0; j < nKFsMpt; ++j) {
            auto& pKF = vpKFsMPt[j];        
            // check whether the map point is observed by the target keyframe
            if (!pMPt->isObservedBy(pKF)) {
                continue;
            }
            // form an edge
            bHasEdge = true;
            cv::KeyPoint kpt = pMPt->keypoint(pKF);
            Eigen::Matrix<double, 2, 1> obs;
            obs << kpt.pt.x, kpt.pt.y;
            g2o::EdgeSE3ProjectXYZ* pEdge = new g2o::EdgeSE3ProjectXYZ();
            // vertex 0: map point
            pEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 pVPt));
            // vertex 1: pose
            pEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 optimizer.vertex(pKF->index())));
            pEdge->setMeasurement(obs);
            // set element in information matrix (value = 1 / sigma^2)
            float sigma = std::pow(Config::scaleFactor(), kpt.octave);
            float invSigma2 = 1.0f / (sigma * sigma);
            pEdge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
            // set robust kernel
            if (bRobust) {
                g2o::RobustKernelHuber* pRK = new g2o::RobustKernelHuber();
                pEdge->setRobustKernel(pRK);
                // rho(x) = x^2 if |x| < delta else 2*delta*|x| - delta^2
                pRK->setDelta(sigma);
            }
            // set cam intrinsics
            pEdge->fx = Config::fx();
            pEdge->fy = Config::fy();
            pEdge->cx = Config::cx();
            pEdge->cy = Config::cy();
            // record all the edges
            vvpEdges[i][j] = pEdge;
            // only optimize outliers at the last iteration
            //if (pMPt->isOutlier()) {
            //    pEdge->setLevel(1);
            //}            
            optimizer.addEdge(pEdge);
        }
        if (!bHasEdge) {
            optimizer.removeVertex(pVPt);
            vbMPtOptimized[i] = false;
        }
    }

    // optimize
    int nIt = 4;
    for (int it = 0; it < nIt; ++it) {
        optimizer.initializeOptimization(0);
        optimizer.optimize(nIter);
        // exclude outliers
        for (int i = 0; i < nMPts; ++i) {
            auto& pMPt = vpMPts[i];
            if (vbMPtOptimized[i]) {
                vector<shared_ptr<KeyFrame>> vpKFsMPt = pMPt->getRelatedKFs();
                int nKFsMpt = vpKFsMPt.size();
                for (int j = 0; j < nKFsMpt; ++j) {
                    auto& pEdge = vvpEdges[i][j];
                    if (!pEdge) {
                        continue;
                    }
                    // optimize all edges for the last iteration
                    // exclude outliers for other iterations
                    if (it == nIt - 2) {
                        pEdge->setLevel(0);
                    } else {
                        float chi2 = pEdge->chi2();
                        cv::KeyPoint kpt = pMPt->keypoint(vpKFsMPt[j]);
                        float sigma = std::pow(Config::scaleFactor(),
                                               kpt.octave);
                        if (chi2 > sigma*sigma ||
                            !pEdge->isDepthPositive()) {
                            pEdge->setLevel(1);
                        } else {
                            pEdge->setLevel(0);
                        }
                    }
                }
            }
        }
    }
    
    // update results back to the keyframes/map
    // pose update via spKFs
    for (const auto& pKF : spKFs) {
        g2o::VertexSE3Expmap* pVSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(pKF->index()));
        g2o::SE3Quat T = pVSE3->estimate();
        pKF->mPose.setPose(SE3Quat2cvMat(T));
    }
    // map point data update via vpMPts
    // count the number of map point outliers after optimization
    for (int i = 0; i < nMPts; ++i) {
        if (vbMPtOptimized[i]) {
            shared_ptr<MapPoint>& pMPt = vpMPts[i];
            g2o::VertexSBAPointXYZ* pVPt = dynamic_cast<
                g2o::VertexSBAPointXYZ*>(optimizer.vertex(idxKFMax + 1 + i));
            Eigen::Vector3d X = pVPt->estimate();
            pMPt->setX3D(Vector3d2cvMat(X));
            
            // set outlier status for all map points
            pMPt->setOutlier(false);
            vector<shared_ptr<KeyFrame>> vpKFsMPt = pMPt->getRelatedKFs();
            int nKFsMpt = vpKFsMPt.size();
            for (int j = 0; j < nKFsMpt; ++j) {
                auto& pEdge = vvpEdges[i][j];
                if (!pEdge) {
                    continue;
                }
                float chi2 = pEdge->chi2();
                cv::KeyPoint kpt = pMPt->keypoint(vpKFsMPt[j]);
                float sigma = std::pow(Config::scaleFactor(), kpt.octave);
                if (chi2 > sigma*sigma * TH_MAX_CHI2_FACTOR ||
                    !pEdge->isDepthPositive()) {
                    pMPt->setOutlier(true);
                    break;
                }
            }            
        }
    }

    // clean up resources
    optimizer.clear();
    optimizer.clearParameters();
}

g2o::SE3Quat Optimizer::cvMat2SE3Quat(const cv::Mat& Tcw) const
{
    // pose data must be double for g2o to use!!!
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 3, 1> t;
    cv::cv2eigen(Tcw.colRange(0, 3).rowRange(0, 3), R); // float -> double
    cv::cv2eigen(Tcw.col(3), t);
    return g2o::SE3Quat(R, t);
}

cv::Mat Optimizer::SE3Quat2cvMat(const g2o::SE3Quat& T) const
{
    Mat Tcw(3, 4, CV_32FC1);
    // Eigen::Quaternion -> Eigen::Matrix3 for R
    Eigen::Matrix<float, 3, 3> R =
        T.rotation().toRotationMatrix().cast<float>(); 
    Eigen::Matrix<float, 3, 1> t = T.translation().cast<float>();
    cv::eigen2cv(R, Tcw.colRange(0, 3).rowRange(0, 3));
    cv::eigen2cv(t, Tcw.col(3));
    return Tcw;
}

Eigen::Vector3d Optimizer::cvMat2Vector3d(const cv::Mat& X3D) const
{
    Eigen::Matrix<double, 3, 1> X;
    cv::cv2eigen(X3D, X);
    return X;
}

cv::Mat Optimizer::Vector3d2cvMat(const Eigen::Vector3d& X) const
{
    Mat X3D(3, 1, CV_32FC1);
    Eigen::Vector3f Xf = X.cast<float>();
    cv::eigen2cv(Xf, X3D);
    return X3D;
}

} // namespace SLAM_demo
