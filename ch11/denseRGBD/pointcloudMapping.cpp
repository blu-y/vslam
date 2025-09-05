//
// Created by Blu on 25. 9. 5..
//

#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

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

    // merge the point clouds
    // intrinsics
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "converting image to point cloud ..." << endl;

    // use XYZRGB as our format
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr pointCloud(new PointCloud);
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        cout << "converting " << i + 1 << endl;
        Mat color = colorImgs[i];
        Mat depth = depthImgs[i];
        Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth data
                if (d == 0) continue; // 0 means invalid reading
                Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];
                current->points.push_back(p);
            }
        // depth filter and statistical removal
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;
    cout << "we have " << pointCloud->size() << " points." << endl;

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.03;
    voxel_filter.setLeafSize(resolution, resolution, resolution);
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    cout << "now we have " << pointCloud->size() << " points after voxel filtering" << endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;
}