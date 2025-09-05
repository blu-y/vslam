//
// Created by Blu on 25. 9. 5..
//

#include <fstream>
#include <iostream>
#include <string>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace Eigen;
using Sophus::SE3d;

/**********************************
 * This program demonstrates the dense depth estimation of a monocular camera
      under a known trajectory use epipolar search + NCC matching method (ch 11.2)
**********************************/


//--------------------------------
// parameters
const int border = 20; // image border
const int width = 640; // image width
const int height = 480; // image height
const double fx = 481.2f; // camera intrinsics
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3; // half window size of NCC
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // area of NCC
const double min_cov = 0.1; // converge criteria: minimal covariance
const double max_cov = 10; // disconverge criteria: maximal covariance


//--------------------------------
// important functions
/// read data from the REMODE dataset
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    Mat &ref_depth);

/**
 * update depth estimate based on a new image
 * @param ref       reference image
 * @param curr      current image
 * @param T_C_R     matrix from ref to cur
 * @param depth     depth estimation
 * @param depth_cov covariance of depth
 * @return          true if success, false otherwise
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2);

/**
 * epipolar search
 * @param ref       reference image
 * @param curr      current image
 * @param T_C_R     matrix from ref to cur
 * @param pt_ref    point in ref
 * @param depth_mu  mean of depth
 * @param depth_cov cov of depth
 * @param pt_curr   point in current
 * @param epipolar_direction    epipolar line direction
 * @return          true if success, false otherwise
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction);

/**
 * update depth filter
 * @param pt_ref    point in ref
 * @param pt_curr   point in cur
 * @param T_C_R     matrix from ref to cur
 * @param epipolar_direction    epipolar line direction
 * @param depth     mean of depth
 * @param depth_cov2    cov of depth
 * @return          true if success, false otherwise
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2);

/**
 * NCC computation
 * @param ref       reference image
 * @param curr      current image
 * @param pt_ref    reference pixel
 * @param pt_curr   current pixel
 * @return          NCC score
 */
double NCC(
    const Mat &ref,
    const Mat &curr,
    const Vector2d &pt_ref,
    const Vector2d &pt_curr);

/// bilinear interpolation
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * static_cast<double>(d[0]) +
        xx * (1 - yy) * static_cast<double>(d[1]) +
        (1 - xx) * yy * static_cast<double>(d[img.step]) +
        xx * yy * static_cast<double>(d[img.step + 1])) / 255.0;
}


//--------------------------------
// some small tools
/// Display the estimated deoth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

/// pixel to camera coordinate system
inline Vector3d px2cam(const Vector2d px) {
    return {(px(0, 0) - cx) / fx,
               (px(1, 0) - cy) / fy,
               1};
}

/// camera coordinate system to pixel
inline Vector2d cam2px(const Vector3d p_cam) {
    return {p_cam(0, 0) * fx / p_cam(2, 0) + cx,
               p_cam(1, 0) * fy / p_cam(2, 0) + cy};
}

/// check if a point is within the image border
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= border
        && pt(1, 0) >= border
        && pt(0, 0) + border < width
        && pt(1, 0) + border <= height;
}

/// show epipolar match
void showEpipolarMatch(
    const Mat &ref,
    const Mat &curr,
    const Vector2d &px_ref,
    const Vector2d &px_curr);

/// show epipolar line
void showEpipolarLine(
    const Mat &ref,
    const Mat &curr,
    const Vector2d &px_ref,
    const Vector2d &px_min_curr,
    const Vector2d &px_max_curr);

/// Evaluate depth estimation
void evaluateDepth(
    const Mat &depth_truth,
    const Mat &depth_estimate);


//--------------------------------
int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "Usage: denseMapping <path_to_test_dataset>" << endl;
    //     return -1;
    // }
    // string dataset_path = argv[1];
    string dataset_path = "../../denseMono/test_data";

    // read data
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(dataset_path, color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // first image
    Mat ref = imread(color_image_files[0], 0); // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0; // initial depth
    double init_cov2 = 3.0; // initial covariance
    Mat depth(height, width, CV_64F, init_depth); // depth image
    Mat depth_cov2(height, width, CV_64F, init_cov2); // depth cov image

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaluateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done. " << endl;

    return 0;
}

bool readDatasetFiles(const string &path, vector<string> &color_image_files, vector<SE3d> &poses, Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // Data format: image_file_name, tx, ty, tz, qx, qy, qz, qw, note that it's TWC not TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                   Vector3d(data[0], data[1], data[2])));
        if (!fin.good()) break;
    }
    fin.close();

    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}

// update the entire depth map
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = border; x < width - border; x++)
        for (int y = border; y < height - border; y++) {
            if (depth_cov2.ptr<double>(y)[x] < min_cov
                || depth_cov2.ptr<double>(y)[x] > max_cov) continue; // converge or abort
            // search match of (x,y) along the epipolar line
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction);
            if (ret == false) continue; // failed

            // to display the match result, uncomment.
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // update if succeed
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    return true;
}

bool epipolarSearch(const Mat &ref, const Mat &curr, const SE3d &T_C_R, const Vector2d &pt_ref, const double &depth_mu, const double &depth_cov, Vector2d &pt_curr, Vector2d &epipolar_direction) {
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu; // reference vector

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // pixel according to mean depth
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); // pixel of minimal depth
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max)); // pixel of maximap depth

    Vector2d epipolar_line = px_max_curr - px_min_curr; // epipolar line
    epipolar_direction = epipolar_line; // normalized
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();
    if (half_length > 100) half_length = 100; // we don't want to search too much

    // to display the epipolar line, uncomment.
    // showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);

    // epipolar search
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;
        if (!inside(px_curr)) continue;
        // compute NCC score
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f) return false;// only trust NCC with high scores
    pt_curr = best_px_curr;
    return true;
}

double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // zero-mean NCC
    // first, compute the mean
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // mean of the reference and current frames
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = double(ref.ptr<uchar>
                (int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;
            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // calculate zero-mean NCC
    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        denominator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        denominator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(denominator1 * denominator2 + 1e-10); // prevent division by zero
}

bool updateDepthFilter(const Vector2d &pt_ref, const Vector2d &pt_curr, const SE3d &T_C_R, const Vector2d &epipolar_direction, Mat &depth, Mat &depth_cov2) {
    // triangulation
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // Equation
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // convert to this:
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;       // result from the ref side
    Vector3d xn = t + ans[1] * f2;      // result from the cur side
    Vector3d p_esti = (xm + xn) / 2.0;  // position of P, taking the average
    double depth_estimation = p_esti.norm(); // depth value

    // compute the covariance
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // Gaussian fusion
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;
    double ave_depth_error_sq = 0;
    int cnt_depth_data = 0;
    for (int y = border; y < depth_truth.rows - border; y++)
        for (int x = border; x < depth_truth.cols - border; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error = " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cvtColor(ref, ref_show, COLOR_GRAY2BGR);
    cvtColor(curr, curr_show, COLOR_GRAY2BGR);

    circle(ref_show, Point2f(px_ref(0, 0), px_ref(1, 0)), 5, Scalar(0, 0, 250), 2);
    circle(curr_show, Point2f(px_curr(0, 0), px_curr(0, 1)), 5, Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr, const Vector2d &px_max_curr) {
    Mat ref_show, curr_show;
    cvtColor(ref, ref_show, COLOR_GRAY2BGR);
    cvtColor(curr, curr_show, COLOR_GRAY2BGR);

    circle(ref_show, Point2f(px_ref(0, 0), px_ref(1, 0)), 5, Scalar(0, 250, 0), 2);
    circle(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, Scalar(0, 250, 0), 2);
    circle(curr_show, Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, Scalar(0, 250, 0), 2);
    line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
        Scalar(0, 250, 0), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}




