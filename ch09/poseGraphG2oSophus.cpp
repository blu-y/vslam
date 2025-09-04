//
// Created by Blu on 25. 9. 4..
//

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <Sophus/se3.hpp>

using namespace std;
using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 1> Vector6d;

// inverse of J in SE(3)
Matrix6d JRInv(const Sophus::SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = Sophus::SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = Sophus::SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = Sophus::SO3d::hat(e.so3().log());
    J = J * 0.5 + Matrix6d::Identity();
    return J;
}

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3d> {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++) is >> data[i];
        setEstimate(Sophus::SE3d(
            Quaterniond(data[6], data[3], data[4], data[5]),
            Vector3d(data[0], data[1], data[2])));
    }

    virtual bool write(ostream &os) const override {
        os << id() << " ";
        Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " "
           << q.coeffs()[2] << " " << q.coeffs()[3] << " ";
        return true;
    }

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    // left multiplication for update
    virtual void oplusImpl(const double *v) override {
        Vector6d upd;
        upd << v[0], v[1], v[2], v[3], v[4], v[5];
        _estimate = Sophus::SE3d::exp(upd) * _estimate;
    }
};

class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++) is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        setMeasurement(Sophus::SE3d(q, Vector3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    virtual bool write(ostream &os) const override {
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";
        Sophus::SE3d m = _measurement;
        Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " "
           << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = 0; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    virtual void computeError() override {
        Sophus::SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        Sophus::SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    // Jacobians
    virtual void linearizeOplus() override {
        Sophus::SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        Sophus::SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(Sophus::SE3d::exp(_error));
        // To do: try to approximate J with I?
        _jacobianOplusXi = -J * v2.inverse().Adj();
        _jacobianOplusXj = J * v1.inverse().Adj();
    }
};

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "Usage: ./poseGraphG2oSE3 sphere.g2o" << endl;
    //     return 1;
    // }
    // string filename = argv[1];
    string filename = "../sphere.g2o";
    ifstream fin(filename);
    if (!fin) {
        cout << "file " << filename << " not found" << endl;
        return 1;
    }

    // setting up g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        make_unique<BlockSolverType>(make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    int vertexCnt = 0, edgeCnt = 0;

    vector<VertexSE3LieAlgebra *> vertices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // built-in SE3 vertex in g2o
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vertices.push_back(v);
            if (index == 0) v-> setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 edges
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges" << endl;

    cout << "optimizing..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results..." << endl;

    ofstream fout("result.g2o");
    for (VertexSE3LieAlgebra *v : vertices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e : edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}