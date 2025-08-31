//
// Created by Blu on 25. 8. 31..
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void pose_estimation_2d2d(
    vector<KeyPoint> keypoints1,
    vector<KeyPoint> keypoints2,
    vector<DMatch> matches,
    Mat &R, Mat &t);
void find_feature_matches(
    const Mat &img1,
    const Mat &img2,
    vector<KeyPoint> &keypoints1,
    vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches);
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    // if (argc != 3) {
    //     cout << "usage: pose_estimation_2d2d img1 img2" << endl;
    //     return 1;
    // }

    // -- Fetch images
    // Mat img1 = imread(argv[1], IMREAD_COLOR);
    // Mat img2 = imread(argv[2], IMREAD_COLOR);
    Mat img1 = imread("../1.png", IMREAD_COLOR);
    Mat img2 = imread("../2.png", IMREAD_COLOR);
    assert(img1.data && img2.data && "Cannot load  images!");

    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
    find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
    cout << "In total, we get " << matches.size() << " set of feature points" << endl;

    // -- Estimate the motion between two frames
    Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);

    // -- Check E=t^R*scale
    Mat t_x = (Mat_<double>(3, 3) <<
        0, -t.at<double>(2, 0), t.at<double>(1, 0),
        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
        -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout << "t^R=" << endl << t_x * R << endl;

    // -- Check epipolar constraints
    Mat K = (Mat_<double>(3, 3) <<
        520.9, 0, 325.1,
        0, 521.0, 249.7,
        0, 0, 1);
    for (DMatch m : matches) {
        Point2d pt1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, 1, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, 1, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
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
    return {
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    };
}