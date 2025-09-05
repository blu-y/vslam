//
// Created by Blu on 25. 9. 5..
//

#include <iostream>
#include <DBoW3/Vocabulary.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    // read the image
    cout << "reading images..." << endl;
    vector<Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "../data/" + to_string(i + 1) + ".png";
        images.push_back(imread(path));
    }
    // detect ORB features
    cout << "detecting ORB features ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<Mat> descriptors;
    for (Mat &image:images) {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // create vocabulary
    cout << "creating vocabulary ..." << endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save("vocabulary.yml.gz");
    cout << "done" << endl;

    return 0;
}