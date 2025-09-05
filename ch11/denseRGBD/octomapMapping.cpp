//
// Created by Blu on 25. 9. 5..
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <octomap/octomap.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <Eigen/src/Geometry/Transform.h>

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char **argv) {
    vector<Mat> colorImgs, depthImgs;
    vector<Isometry3d> poses;

    ifstream fin("../../denseRGBD/data/pose.txt");
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("../../denseRGBD/data/%s/%d.%s");
        colorImgs.push_back(imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(imread((fmt % "depth" % (i + 1) % "png").str(), -1));

        double data[7] = {0};
        for (int i = 0; i < 7; i++) fin >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        Isometry3d T(q);
        T.pretranslate(Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // intrinsics
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "converting image to octomap ..." << endl;

    // octomap tree
    octomap::OcTree tree(0.01); // resolution
    for (int i = 0; i < 5; i++) {
        cout << "Converting " << i + 1 << endl;
        Mat color = colorImgs[i];
        Mat depth = depthImgs[i];
        Isometry3d T = poses[i];

        octomap::Pointcloud cloud; // the pointcloud in octomap

        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth data
                if (d == 0) continue; // 0 means invalid reading
                Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Vector3d pointWorld = T * point;
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }

        // save into octo tree
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3),
                                                                 T(1, 3),
                                                                 T(2, 3)));
    }

    // update and save into a bt file
    tree.updateInnerOccupancy();
    cout << "saving octomap ..." << endl;
    tree.writeBinary("octomap.bt");

    octomap::Pointcloud pc;
    for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
        if (tree.isNodeOccupied(*it)) {
            pc.push_back(it.getX(), it.getY(), it.getZ());
        }
    }
    pc.writeVrml("octomap_points.wrl");   // VRML 파일로 저장


    return 0;
}