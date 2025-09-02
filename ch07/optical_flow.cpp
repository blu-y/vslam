//
// Created by Blu on 25. 9. 1..
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2,
    const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
    vector<bool> &success, bool inverse = false, bool has_initial = false);

void OpticalFlowMultiLevel(const Mat &img1, const Mat &img2,
    const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
    vector<bool> &success, bool inverse);

inline float GetPixelValue(const Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = min(img.cols - 1, int(x) + 1);
    int y_a1 = min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv) {

    // images are grayscale CV_8UC1
    Mat img1 = imread("../LK1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("../LK2.png", IMREAD_GRAYSCALE);
    assert(img1.data && img2.data && "Cannot load images!");

    // keypoints, using GFTT
    vector<KeyPoint> kp1;
    Ptr<FeatureDetector> detector = GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);

    // track these keypoints in second image
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << " seconds." << endl;

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << " seconds." << endl;

    // plot the differences
    Mat img2_single;
    cvtColor(img2, img2_single, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            circle(img2_single, kp2_single[i].pt, 2, Scalar(0, 250, 0), 2);
            line(img2_single, kp1[i].pt, kp2_single[i].pt, Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cvtColor(img2, img2_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            circle(img2_multi, kp2_multi[i].pt, 2, Scalar(0, 250, 0), 2);
            line(img2_multi, kp1[i].pt, kp2_multi[i].pt, Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cvtColor(img2, img2_CV, COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            circle(img2_CV, pt2[i], 2, Scalar(0, 250, 0), 2);
            line(img2_CV, pt1[i], pt2[i], Scalar(0, 250, 0), 2);
        }
    }

    imshow("tracked single level", img2_single);
    imshow("tracked multi level", img2_multi);
    imshow("tracked by opencv", img2_CV);
    waitKey(0);

    return 0;
}


class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true,
        bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_),
        success(success_), inverse(inverse_), has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx, dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        Matrix2d H = Matrix2d::Zero(); // Hessian
        Vector2d b = Vector2d::Zero(); // bias
        Vector2d J; // Jacobian
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                H = Matrix2d::Zero();
                b = Vector2d::Zero();
            } else {
                // only reset b
                b = Vector2d::Zero();
            }

            cost = 0;

            // compute cost and Jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy); // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y)
                                     - GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1)
                                     - GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated,
                        // so we can store it and only compute error
                        J = -1.0 * Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y)
                                     - GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1)
                                     - GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)));
                    }
                    // compute H, b and set cost
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // compute update
            Vector2d update = H.ldlt().solve(b);

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) break;

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) break; // converge
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2,
    const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
    vector<bool> &success, bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()),
        bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}


void OpticalFlowMultiLevel(const Mat &img1, const Mat &img2,
    const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
    vector<bool> &success, bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            Mat img1_pyr, img2_pyr;
            resize(pyr1[i - 1], img1_pyr,
                Size(pyr1[i - 1].cols * pyramid_scale,
                          pyr1[i - 1].rows * pyramid_scale));
            resize(pyr2[i - 1], img2_pyr,
                Size(pyr2[i - 1].cols * pyramid_scale,
                          pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level],
            kp1_pyr, kp2_pyr,
            success, inverse, true);

        if (level > 0) {
            for (auto &kp:kp1_pyr) kp.pt /= pyramid_scale;
            for (auto &kp:kp2_pyr) kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp:kp2_pyr) kp2.push_back(kp);
}