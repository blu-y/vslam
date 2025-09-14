//
// Created by Blu on 25. 9. 11..
//

#pragma once
#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include "mySLAM/common_include.h"
#include "mySLAM/frame.h"
#include "mySLAM/map.h"

#include <opencv2/features2d.hpp>

namespace  mySLAM {

    class Backend;
    class Viewer;

    enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST };

    /**
     * Frontend
     * Estimates pose of current frame,
     * Adds keyframes to map and triggers optimization.
     * (when keyframe conditions are met)
     */
    class Frontend {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<Frontend> Ptr;

        Frontend();

        /// External interface, adds a frame and computes its localization result.
        bool AddFrame(Frame::Ptr frame);

        /// Set functions
        void SetMap(Map::Ptr map) { map_ = map; }
        void SetBackend(shared_ptr<Backend> backend) { backend_ = backend; }
        void SetViewer(shared_ptr<Viewer> viewer) { viewer_ = viewer; }

        FrontendStatus GetStatus() { return status_; }

        void SetCameras(Camera::Ptr left, Camera::Ptr right) {
            camera_left_ = left;
            camera_right_ = right;
        };

    private:
        /**
         * Track in normal mode
         * @return true if success
         */
        bool Track();

        /**
         * Reset when lost
         * @return true if success
         */
        bool Reset();

        /**
         * Track with last frame
         * @return num of tracked points
         */
        int TrackLastFrame();

        /**
         * estimate current frame's pose
         * @return num of inliers
         */
        int EstimateCurrentPose();

        /**
         * set current frame as a keyframe and insert it into backend
         * @return true if success
         */
        bool InsetKeyframe();

        /**
         * Try init the frontend with stereo images saved in current_frame_
         * @return true if success
         */
        bool StereoInit();

        /**
         * Detect features in left image in current_frame_
         * keypoints will be saved in current_frame_
         * @return num of detected features
         */
        int DetectFeatures();

        /**
         * Find the corresponding features in right image of current_frame_
         * @return num of features found
         */
        int FindFeaturesInRight();

        /**
         * Build the initial map with single image
         * @return true if succeed
         */
        bool BuildInitMap();

        /**
         * Triangulate the 2D points in current frame
         * @return num of triangulated points
         */
        int TriangulateNewPoints();

        /**
         * Set the features in keyframe as new observation of the map points
         */
        void SetObservationsForKeyFrame();

        // data
        FrontendStatus status_ = FrontendStatus::INITING;

        Frame::Ptr current_frame_ = nullptr;
        Frame::Ptr last_frame_ = nullptr;
        Camera::Ptr camera_left_ = nullptr;
        Camera::Ptr camera_right_ = nullptr;

        Map::Ptr map_ = nullptr;
        shared_ptr<Backend> backend_ = nullptr;
        shared_ptr<Viewer> viewer_ = nullptr;

        /// relative motion between current and last frame
        /// used to estimate the initial pose of the current frame
        SE3 relative_motion_;

        int tracking_inliers_ = 0; // inliers, used for testing new kf

        // params
        int num_features_ = 200;
        int num_features_init_ = 100;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;

        // utilities
        cv::Ptr<cv::GFTTDetector> gftt_; // feature detector in opencv
    };

}

#endif //MYSLAM_FRONTEND_H