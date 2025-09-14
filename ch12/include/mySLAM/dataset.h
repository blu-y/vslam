//
// Created by Blu on 25. 9. 10..
//

#pragma once
#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H

#include "frame.h"
#include "mySLAM/common_include.h"
#include "mySLAM/frame.h"

namespace mySLAM {

    /**
     * Dataset reading
     * Pass the configuration file path during construction.
     * The configuration file's `dataset_dir` is the dataset path.
     * After `init`, the camera and next frame image are available.
     */
    class Dataset {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<Dataset> Ptr;
        Dataset(const string &dataset_path);

        /// Initialization, true if successful
        bool Init();

        /// create and return the next frame containing the stereo images
        Frame::Ptr NextFrame();

        /// get camera by id
        Camera::Ptr GetCamera(int camera_id) const {
            return cameras_.at(camera_id);
        }

    private:
        string dataset_path_;
        int current_image_index_ = 0;

        vector<Camera::Ptr> cameras_;
    };

}

#endif //MYSLAM_DATASET_H