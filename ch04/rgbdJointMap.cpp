//
// Created by Blu on 25. 8. 28..
//


#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace Sophus;

typedef vector<SE3d, Eigen::aligned_allocator<SE3d>> TrajectoryType;
typedef vector<Vector6d, Eigen::aligned_allocator<Vector6d>> PointCloudType;

void showPointCloud(const PointCloudType &pointcloud);

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;
    TrajectoryType poses; // camera poses

    ifstream fin("../rgbd/pose.txt");
    if (!fin) {
        cerr << "pose.txt not found" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("../rgbd/%s/%d.%s"); // the image format
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // use -1 flag to load the depth image

        double data[7] = {0};
        for (auto &d: data) fin >> d;
        SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }

    // compute the point cloud using camera intrinsics
    double cx = 325.5, cy = 253.5, fx = 518.0, fy = 519.0;
    double depthScale = 1000.0;
    PointCloudType pointcloud;
    pointcloud.reserve(1000000);

    for (int i = 0; i < 5; i++) {
        cout << "Converting RGBD images " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth value is 16-bit
                if (d == 0) continue; // 0 means no valid value
                Eigen::Vector3d point;
                point[2] = static_cast<double>(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()]; // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
        }
    }

    cout << "global point cloud has " << pointcloud.size() << " points." << endl;
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const PointCloudType &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "point cloud is empty" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Pointcloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
        );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3] / 255.0, p[4] / 255.0 , p[5] / 255.0); // depth as color
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);
    }
}