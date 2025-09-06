//
// Created by Blu on 25. 9. 6..
//

#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "mySLAM/camera.h"
#include "mySLAM/common_include.h"

namespace mySLAM {

    // forward declare
    struct MapPoint;
    struct Feature;

/**
 * each frame gets its own frame id,
 * each keyframe gets its own keyframe id
 */
    struct Frame {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id_ = 0;          // id of this frame
        unsigned long keyframe_id_ = 0; // id of keyframe
        bool is_keyframe_ = false;      // true if keyframe
        double time_stamp_;             // timestamp
        SE3 pose_;                      // pose defined as Tcw
        mutex pose_mutex_;         // pose data mutex
        Mat left_img_, right_img_;      // stereo images

        // extracted features in left image
        vector<shared_ptr<Feature>> features_left_;
        // corresponding features in right image
        // set to nullptr if no corresponding
        vector<shared_ptr<Feature>> features_right_;

    public: // data members
        Frame() {}

        Frame(long id, double time_stamp, const SE3 &pose,
              const Mat &left, const Mat &right);

        // set and get pose, thread safe
        SE3 Pose() {
            unique_lock<mutex> lck(pose_mutex_);
            return pose_;
        }

        void SetPose(const SE3 &pose) {
            unique_lock<mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        /// Set keyframe and keyframe_id
        void SetKeyFrame();

        /// create new frame and allocate id
        static shared_ptr<Frame> CreateFrame();
    };

}

#endif //MYSLAM_FRAME_H