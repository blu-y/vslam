//
// Created by Blu on 25. 9. 1..
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Sophus/se3.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t);
void bundleAdjustment(const vector<Point3f> &pts1,
                      const vector<Point3f> &pts2,
                      Mat &R, Mat &t);
void find_feature_matches(const Mat &img1, const Mat &img2,
                          vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
                          vector<DMatch> &matches);
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {

    Mat img1 = imread("../1.png", IMREAD_COLOR);
    Mat img2 = imread("../2.png", IMREAD_COLOR);
    Mat depth1 = imread("../1_depth.png", IMREAD_UNCHANGED);
    Mat depth2 = imread("../2_depth.png", IMREAD_UNCHANGED);

    assert(img1.data && img2.data && "Cannot load images!");

    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
    cout << "matches: " << matches.size() << endl;

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1;
    vector<Point3f> pts2;

    for (DMatch m : matches) {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints2[m.trainIdx].pt.y))[int(keypoints2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) continue; // bad depth
        float dd1 = d1 / 5000.0;
        float dd2 = d2 / 5000.0;
        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints2[m.trainIdx].pt, K);
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }
    cout << "3d-3d pairs: " << pts1.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "ICP via SVD results: " << time_span.count() << " seconds." << endl;
    cout << "R= " << endl << R << endl;
    cout << "t= " << endl << t << endl;
    cout << "R_inv= " << endl << R.t() << endl;
    cout << "t_inv= " << endl << -R.t() * t << endl;

    cout << "calling bundle_adjustment" << endl;

    bundleAdjustment(pts1, pts2, R, t);

    // verify p1 = R * p2 + t
    for (int i = 0; i < 5; i++) {
        cout << "p1= " << pts1[i] << endl;
        cout << "p2= " << pts2[i] << endl;
        cout << "(R*p2+t)= " <<
            R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
            << endl;
        cout << endl;
    }
    return 0;
}

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
    Point3f p1, p2; // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute W = sum(q1*q2^T)
    Matrix3d W = Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Vector3d(q1[i].x, q1[i].y, q1[i].z) * Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W= " << W << endl;

    // SVD on W, W = UΣV^T
    JacobiSVD<Matrix3d> svd(W, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();
    cout << "U= " << U << endl;
    cout << "V= " << V << endl;

    Matrix3d R_ = U * V.transpose();
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Vector3d t_ = Vector3d(p1.x, p1.y, p1.z) - R_ * Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *v) override {
        Matrix<double, 6, 1> update_eigen;
        update_eigen << v[0], v[1], v[2], v[3], v[4], v[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &is) override {}
    virtual bool write(ostream &os) const override {}
};

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Vector3d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectXYZRGBDPoseOnly(const Vector3d &point) : _point(point) {}

    virtual void computeError() override {
        const VertexPose* pose = static_cast<const VertexPose*>(_vertices[0]);
        _error = _measurement - pose->estimate() * _point;
    }

    virtual void linearizeOplus() override {
        const VertexPose* pose = static_cast<const VertexPose*>(_vertices[0]);
        Sophus::SE3d T = pose->estimate();
        Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }

    bool read(istream &in) {}
    bool write(ostream &out) const {}

private:
    Vector3d _point;
};

void bundleAdjustment(const vector<Point3f> &pts1,
                      const vector<Point3f> &pts2,
                      Mat &R, Mat &t) {
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexPose *vertex = new VertexPose();
    vertex->setId(0);
    vertex->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex);

    for (int i = 0; i < pts1.size(); i++) {
        Vector3d p1(pts1[i].x, pts1[i].y, pts1[i].z);
        Vector3d p2(pts2[i].x, pts2[i].y, pts2[i].z);
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(p2);
        edge->setId(i);
        edge->setVertex(0, vertex);
        edge->setMeasurement(p1);
        edge->setInformation(Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    cout << "start optimization" << endl;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> diff = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "optimization time: " << diff.count() << " seconds." << endl;
    cout << "pose estimation by g2o =\n" << vertex->estimate().matrix() << endl;

    Matrix3d R_ = vertex->estimate().rotationMatrix();
    Vector3d t_ = vertex->estimate().translation();
    R = (Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

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