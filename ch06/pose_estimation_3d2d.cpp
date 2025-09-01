//
// Created by Blu on 25. 8. 31..
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Sophus/se3.hpp>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace cv;
using namespace Eigen;

typedef vector<Vector2d, aligned_allocator<Vector2d>> VecVector2d;
typedef vector<Vector3d, aligned_allocator<Vector3d>> VecVector3d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;

void find_feature_matches(const Mat &img1, const Mat &img2,
                          vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
                          vector<DMatch> &matches);
Point2d pixel2cam(const Point2d& p, const Mat& K);
void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d,
                         const Mat &K, Sophus::SE3d &pose);
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d,
                                 const Mat &K, Sophus::SE3d &pose);

int main(int argc, char **argv) {

    Mat img1 = imread("../1.png", IMREAD_COLOR);
    Mat img2 = imread("../2.png", IMREAD_COLOR);
    Mat depth1 = imread("../1_depth.png", IMREAD_UNCHANGED);

    assert(img1.data && img2.data && "Cannot load images!");

    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
    cout << "matches: " << matches.size() << endl;

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for (DMatch m : matches) {
        ushort d = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        if (d == 0) continue; // bad depth
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // flags Iterative, EPNP, DLS, etc.
    Mat R;
    Rodrigues(r, R); // Rodrigues formula: rotation vector r -> rotation matrix R
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "solve pnp(OpenCV) cost time: " << elapsed.count() << " seconds." << endl;
    cout << "R= " << endl << R << endl;
    cout << "t= " << endl << t << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); i++) {
        pts_3d_eigen.push_back(Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by GN" << endl;
    Sophus::SE3d pose_gn;
    start = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    end = chrono::steady_clock::now();
    elapsed = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "solve pnp(G-N) cost time: " << elapsed.count() << " seconds." << endl;

    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    start = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    end = chrono::steady_clock::now();
    elapsed = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "solve pnp(g2o) cost time: " << elapsed.count() << " seconds." << endl;
    return 0;
}


void find_feature_matches(const Mat &img1, const Mat &img2,
                          vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
                          vector<DMatch> &matches) {
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    vector<DMatch> match;
    matcher->match(descriptors1, descriptors2, match);

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    for (int i = 0; i < descriptors1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d& p, const Mat& K) {
    return {(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
               (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)};
}

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,
                                 const VecVector2d &points_2d,
                                 const Mat &K, Sophus::SE3d &pose) {
    const int iterations = 10;
    double cost = 0, last_cost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) {
        Matrix6d H = Matrix6d::Zero(); // Hessian
        Vector6d b = Vector6d::Zero(); // bias
        cost = 0;
        for (int i = 0; i < points_3d.size(); i++) {
            Vector3d pc = pose * points_3d[i];
            double xi = pc[0], yi = pc[1], zi = pc[2];
            Vector2d proj(fx * xi / zi + cx, fy * yi / zi + cy);
            Vector2d e = points_2d[i] - proj;

            // calculate Jacobian
            Matrix<double, 2, 6> J;
            J << -fx / zi,
                0,
                fx * xi / zi / zi,
                fx * xi * yi / zi / zi,
                -fx - fx * xi * xi / zi / zi,
                fx * yi / zi,
                0,
                -fy / zi,
                fy * yi / zi / zi,
                fy + fy * yi * yi / zi / zi,
                -fy * xi * yi / zi / zi,
                -fy * xi / zi;

            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += e.squaredNorm();
        }

        // solve Hx=b
        Vector6d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter < 0 && cost >= last_cost) {
            // if cost increases, update is not good
            cout << "cost: " << cost << ", last cost: " << last_cost << endl;
            break;
        }

        // update estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        last_cost = cost;

        cout << "iteration " << iter << " cost= " << setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) { // converge
            break;
        }
    }
    cout << "pose by G-N: \n" << pose.matrix() << endl;
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // override the reset function
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
        Vector6d update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjection(const Vector3d &pos, Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0], Y = _pos3d[1], Z = _pos3d[2];
        _jacobianOplusXi << -fx / Z,
                            0,
                            fx * X / Z / Z,
                            fx * X * Y / Z / Z,
                            -fx - fx * X * X / Z / Z,
                            fx * Y / Z,
                            0,
                            -fy / Z,
                            fy * Y / Z / Z,
                            fy + fy * Y * Y / Z / Z,
                            -fy * X * Y / Z / Z,
                            -fy * X / Z;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}

private:
    Vector3d _pos3d;
    Matrix3d _K;
};

void bundleAdjustmentG2O(const VecVector3d &points_3d,
                         const VecVector2d &points_2d,
                         const Mat &K, Sophus::SE3d &pose) {
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // add vertex
    VertexPose *vertex = new VertexPose();
    vertex->setId(0);
    vertex->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex);

    // K
    Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
               K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
               K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // add edges
    for (int i = 0; i < points_3d.size(); i++) {
        EdgeProjection *edge = new EdgeProjection(points_3d[i], K_eigen);
        edge->setId(i);
        edge->setVertex(0, vertex);
        edge->setMeasurement(points_2d[i]);
        edge->setInformation(Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> elapsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << elapsed.count() << " seconds." << endl;
    pose = vertex->estimate();
    cout << "pose estimation by g2o =\n" << pose.matrix() << endl;
}



