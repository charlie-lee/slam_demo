/**
 * @file   Optimizer.cpp
 * @brief  Implementations of Optimizer class for pose & map data optimization.
 * @author Charlie Li
 * @date   2019.09.17
 */

#include "Optimizer.hpp"

#include <memory>
#include <vector>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
//#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h> // g2o::RobustKernelHuber, ...
#include <g2o/core/solver.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h> // for global BA
#include <g2o/solvers/dense/linear_solver_dense.h> // for pose optimization
//#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h> // g2o::EdgeSE3ProjectXYZ, ...
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp> // cv::cv2eigen()
#include <Eigen/Core>
#include <Eigen/Geometry> // Eigen::Quaternion
#include "Config.hpp"
#include "Frame.hpp"
#include "Map.hpp"
#include "MapPoint.hpp"
#include "Tracker.hpp"

namespace SLAM_demo {

using std::shared_ptr;
using std::make_shared;
using std::vector;
using cv::Mat;

Optimizer::Optimizer(const std::shared_ptr<Map>& pMap) : mpMap(pMap) {}

void Optimizer::globalBundleAdjustment(int nIter, bool bRobust) const
{
    // set definition of solver
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver;
    pLinearSolver =
        new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* pBlkSolver;
    pBlkSolver = new g2o::BlockSolver_6_3(pLinearSolver);
    g2o::OptimizationAlgorithmLevenberg* pSolverLM;
    pSolverLM = new g2o::OptimizationAlgorithmLevenberg(pBlkSolver);
    // configure optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    optimizer.setAlgorithm(pSolverLM);
    
    // add vertices: poses
    vector<shared_ptr<Frame>> vpFrames = mpMap->getAllFrames();
    unsigned idxFMax = 0;
    for (const auto& pFrame : vpFrames) {
        g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
        Mat Tcw = pFrame->mPose.getPose();
        unsigned idxF = pFrame->getFrameIdx();
        pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        pVSE3->setId(idxF);
        bool bFixed = idxF == Tracker::n1stFrame;
        pVSE3->setFixed(bFixed);
        
        optimizer.addVertex(pVSE3);
        if (idxF > idxFMax) {
            idxFMax = idxF;
        }
    }
    // add vertices: map points, and add edges for each map point
    vector<shared_ptr<MapPoint>> vpMPts = mpMap->getAllMPts();
    int nMPts = vpMPts.size();
    for (int i = 0; i < nMPts; ++i) {
        const shared_ptr<MapPoint>& pMPt = vpMPts[i];
        g2o::VertexSBAPointXYZ* pVPt = new g2o::VertexSBAPointXYZ();
        pVPt->setEstimate(cvMat2Vector3d(pMPt->getX3D()));
        pVPt->setId(idxFMax + 1 + i);
        pVPt->setMarginalized(true); // why?? (to decrease the size of Hessian?)
        optimizer.addVertex(pVPt);
        
        // add edges
        vector<shared_ptr<Frame>> vpFramesMPt = pMPt->getRelatedFrames();
        for (const auto& pFrame : vpFramesMPt) {
            // is it necessary to check whether the frame is valid?
            cv::KeyPoint kpt = pMPt->getKpt(pFrame);
            Eigen::Matrix<double, 2, 1> obs;
            obs << kpt.pt.x, kpt.pt.y;
            g2o::EdgeSE3ProjectXYZ* pEdge = new g2o::EdgeSE3ProjectXYZ();
            // vertex 0: map point
            pEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 pVPt));
            // vertex 1: pose
            pEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 optimizer.vertex(pFrame->getFrameIdx())));
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
            // ?
            pEdge->setParameterId(0, 0);
            
            optimizer.addEdge(pEdge);
        }
    }
    
    // optimize
    optimizer.initializeOptimization();
    optimizer.optimize(nIter);
    
    // update results back to the frames/map
    // pose update via vpFrames
    for (const auto& pFrame : vpFrames) {
        g2o::VertexSE3Expmap* pVSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(pFrame->getFrameIdx()));
        g2o::SE3Quat T = pVSE3->estimate();
        pFrame->mPose.setPose(SE3Quat2cvMat(T));
    }
    // map point data update via vpMPts
    for (int i = 0; i < nMPts; ++i) {
        shared_ptr<MapPoint>& pMPt = vpMPts[i];
        g2o::VertexSBAPointXYZ* pVPt = dynamic_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(idxFMax + 1 + i));
        Eigen::Vector3d X = pVPt->estimate();
        pMPt->setX3D(Vector3d2cvMat(X));
    }
}

void Optimizer::frameBundleAdjustment(int nIter, bool bRobust) const
{
    // set definition of solver
    g2o::BlockSolver_6_3::LinearSolverType* pLinearSolver;
    pLinearSolver =
        new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* pBlkSolver;
    pBlkSolver = new g2o::BlockSolver_6_3(pLinearSolver);
    g2o::OptimizationAlgorithmLevenberg* pSolverLM;
    pSolverLM = new g2o::OptimizationAlgorithmLevenberg(pBlkSolver);
    // configure optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    optimizer.setAlgorithm(pSolverLM);
    
    // add vertices: poses
    vector<shared_ptr<Frame>> vpFrames = mpMap->getAllFrames();
    unsigned idxFMax = 0;
    for (const auto& pFrame : vpFrames) {
        g2o::VertexSE3Expmap* pVSE3 = new g2o::VertexSE3Expmap();
        Mat Tcw = pFrame->mPose.getPose();
        unsigned idxF = pFrame->getFrameIdx();
        pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        pVSE3->setId(idxF);
        bool bFixed = idxF != System::nCurrentFrame;
        pVSE3->setFixed(bFixed);
        optimizer.addVertex(pVSE3);
        if (idxF > idxFMax) {
            idxFMax = idxF;
        }
    }
    // add vertices: map points, and add edges for each map point
    vector<shared_ptr<MapPoint>> vpMPts = mpMap->getAllMPts();
    int nMPts = vpMPts.size();
    for (int i = 0; i < nMPts; ++i) {
        const shared_ptr<MapPoint>& pMPt = vpMPts[i];
        g2o::VertexSBAPointXYZ* pVPt = new g2o::VertexSBAPointXYZ();
        pVPt->setEstimate(cvMat2Vector3d(pMPt->getX3D()));
        pVPt->setId(idxFMax + 1 + i);
        pVPt->setMarginalized(true); // why?? (to decrease the size of Hessian?)
        // set map point vertex as fixed if it is not newly triangulated
        vector<shared_ptr<Frame>> vpRFrames = pMPt->getRelatedFrames();
        bool bFixed = !(pMPt->getIdxLastObsFrm() == System::nCurrentFrame &&
                        vpRFrames.size() == 2);
        pVPt->setFixed(bFixed);
        optimizer.addVertex(pVPt);
        
        // add edges
        vector<shared_ptr<Frame>> vpFramesMPt = pMPt->getRelatedFrames();
        for (const auto& pFrame : vpFramesMPt) {
            // is it necessary to check whether the frame is valid?
            cv::KeyPoint kpt = pMPt->getKpt(pFrame);
            Eigen::Matrix<double, 2, 1> obs;
            obs << kpt.pt.x, kpt.pt.y;
            g2o::EdgeSE3ProjectXYZ* pEdge = new g2o::EdgeSE3ProjectXYZ();
            // vertex 0: map point
            pEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 pVPt));
            // vertex 1: pose
            pEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                 optimizer.vertex(pFrame->getFrameIdx())));
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
            // ?
            pEdge->setParameterId(0, 0);
            
            optimizer.addEdge(pEdge);
        }
    }
    
    // optimize
    optimizer.initializeOptimization();
    optimizer.optimize(nIter);
    
    // update results back to the frames/map
    // pose update via vpFrames
    for (const auto& pFrame : vpFrames) {
        g2o::VertexSE3Expmap* pVSE3 = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(pFrame->getFrameIdx()));
        g2o::SE3Quat T = pVSE3->estimate();
        pFrame->mPose.setPose(SE3Quat2cvMat(T));
    }
    // map point data update via vpMPts
    for (int i = 0; i < nMPts; ++i) {
        shared_ptr<MapPoint>& pMPt = vpMPts[i];
        g2o::VertexSBAPointXYZ* pVPt = dynamic_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(idxFMax + 1 + i));
        Eigen::Vector3d X = pVPt->estimate();
        pMPt->setX3D(Vector3d2cvMat(X));
    }    
}

void Optimizer::poseOptimization(const std::shared_ptr<Frame>& pFrame) const
{
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
    vector<shared_ptr<MapPoint>> vpMPts = pFrame->getpMPtsObserved();
    int nMPts = vpMPts.size();
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdges;
    vpEdges.reserve(nMPts);
    for (int i = 0; i < nMPts; ++i) {
        const shared_ptr<MapPoint>& pMPt = vpMPts[i];
        // is it necessary to check whether the frame is valid?
        cv::KeyPoint kpt = pMPt->getKpt(pFrame);
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
        pEdge->Xw = cvMat2Vector3d(pMPt->getX3D());
        // set cam intrinsics
        pEdge->fx = Config::fx();
        pEdge->fy = Config::fy();
        pEdge->cx = Config::cx();
        pEdge->cy = Config::cy();
        // ?
        pEdge->setParameterId(0, 0);
        vpEdges.push_back(pEdge);
        optimizer.addEdge(pEdge);
    }
    
    // optimize (multi-pass?)
    int nIt = 4;
    for (int it = 0; it < nIt; ++it) {
        //pVSE3->setEstimate(cvMat2SE3Quat(Tcw));
        optimizer.initializeOptimization(0); // only optimize level 0 edges
        optimizer.optimize(10);
        // exclude outliers
        for (int i = 0; i < nMPts; ++i) {
            auto& pMPt = vpMPts[i];
            auto& pEdge = vpEdges[i];
            // optimize all edges for the last iteration
            // exclude outliers for other iterations
            if (it == nIt - 2) {
                pEdge->setLevel(0);
            } else {
                float chi2 = pEdge->chi2();
                cv::KeyPoint kpt = pMPt->getKpt(pFrame);
                float sigma = std::pow(Config::scaleFactor(), kpt.octave);
                if (chi2 > sigma*sigma) {
                    pEdge->setLevel(1);
                } else {
                    pEdge->setLevel(0);
                }
            }
        }
    }
    
    // update optimized pose and map point outlier status
    for (int i = 0; i < nMPts; ++i) {
        auto& pMPt = vpMPts[i];
        auto& pEdge = vpEdges[i];
        float chi2 = pEdge->chi2();
        cv::KeyPoint kpt = pMPt->getKpt(pFrame);
        float sigma = std::pow(Config::scaleFactor(), kpt.octave);
        if (chi2 > sigma*sigma) {
            pMPt->setOutlier(true);
        } else {
            pMPt->setOutlier(false);
        }
    }
    g2o::SE3Quat T = pVSE3->estimate();
    pFrame->mPose.setPose(SE3Quat2cvMat(T));
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