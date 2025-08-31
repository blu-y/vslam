//
// Created by Blu on 25. 8. 31..
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void triangulation(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2,
                   const vector<DMatch> &matches, const Mat &R, const Mat &t,
                   vector<Point3d> &points);
void pose_estimation_2d2d(vector<KeyPoint> keypoints1,
                          vector<KeyPoint> keypoints2,
                          vector<DMatch> matches, Mat &R, Mat &t);
void find_feature_matches(const Mat &img1, const Mat &img2,
                          vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
                          vector<DMatch> &matches);
Point2d pixel2cam(const Point2d &p, const Mat &K);
inline Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth < up_th) depth = up_th;
    if (depth > low_th) depth = low_th;
    return {255 * depth / th_range, 0, 255 * (1 - depth / th_range)};
}

int main(int argc, char **argv) {
    Mat img1 = imread("../1.png", IMREAD_COLOR);
    Mat img2 = imread("../2.png", IMREAD_COLOR);
    assert(img1.data && img2.data && "Cannot load  images!");

    // find feature & match
    vector<KeyPoint> kp1, kp2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, kp1, kp2, matches);
    cout << "Matches found: " << matches.size() << endl;

    // estimate post
    Mat R, t;
    pose_estimation_2d2d(kp1, kp2, matches, R, t);
    cout << "R: " << R << endl;
    cout << "t: " << t << endl;

    // triangulation
    vector<Point3d> points;
    triangulation(kp1, kp2, matches, R, t, points);

    // Verify the reprojection relationship between triangulated points and feature points
    Mat K = (Mat_<double>(3, 3) <<
        520.9, 0, 325.1,
        0, 521.0, 249.7,
        0, 0, 1);
    Mat img1_plot = img1.clone();
    Mat img2_plot = img2.clone();
    for (int i = 0; i < matches.size(); i++) {
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(kp1[matches[i].queryIdx].pt, K);
        circle(img1_plot, kp1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        circle(img2_plot, kp2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    imshow("img1", img1_plot);
    imshow("img2", img2_plot);
    waitKey(0);

    return 0;
}

void triangulation(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2,
                   const vector<DMatch> &matches, const Mat &R, const Mat &t,
                   vector<Point3d> &points) {
    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), R.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), R.at<double>(2, 0));

    Mat K = (Mat_<double>(3, 3) <<
        520.9, 0, 325.1,
        0, 521.0, 249.7,
        0, 0, 1);
    vector<Point2f> pts1, pts2;
    for (DMatch m:matches) {
        // Convert pixel coordinates to camera coordinates
        pts1.push_back(pixel2cam(kp1[m.queryIdx].pt, K));
        pts2.push_back(pixel2cam(kp2[m.trainIdx].pt, K));
    }

    Mat pts4d;
    triangulatePoints(T1, T2, pts1, pts2, pts4d);

    // Convert to non-homogeneous coordinates
    for (int i = 0; i < pts4d.cols; i++) {
        Mat x = pts4d.col(i);
        x /= x.at<float>(3, 0);
        Point3d p(x.at<float>(0, 0),
                  x.at<float>(1, 0),
                  x.at<float>(2, 0));
        points.push_back(p);
    }
}

void pose_estimation_2d2d(vector<KeyPoint> keypoints1,
                          vector<KeyPoint> keypoints2,
                          vector<DMatch> matches, Mat &R, Mat &t) {
    // Camera Intrinsics, TUM Feiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // -- Convert the matching point to the form of vector<Point2f>
    vector<Point2f> points1, points2;

    for (int i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // -- Calculate fundamental matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
    cout << "fundamental_matrix: " << fundamental_matrix << endl;

    // -- Calculate essential matrix
    Point2d principal_point(325.1, 249.1); // camera principal point, calibrated in TUM dataset
    double focal_length = 521; // camera focal length, calibrated in TUM dataset
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix: " << essential_matrix << endl;

    // -- Calculate homography matrix
    // -- But the scene is not planar, and calculating the homography matrix here is of little significance
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix: " << homography_matrix << endl;

    // -- Recover rotation and translation from the essential matrix
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R: " << endl << R << endl;
    cout << "t: " << endl << t << endl;
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