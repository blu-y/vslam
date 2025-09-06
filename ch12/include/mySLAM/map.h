//
// Created by Blu on 25. 9. 6..
//

#pragma once

#ifndef MYSLAM_MAP_H
#define MYSLAM_MAP_H

#include "mySLAM/common_include.h"
#include "mySLAM/frame.h"
#include "mySLAM/mappoint.h"

namespace mySLAM {

    /**
    * @brief Map
    * The frontend calls InsertKeyframe and InsertMapPoint
    *   to insert new frames and map points;
    * the backend maintains the map structure,
    *   determines outliers, removes them, etc.
    */
    class Map {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<Map> Ptr;
        typedef unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
        typedef unordered_map<unsigned long, Frame::Ptr> KeyframesType;

        Map() {}

        void InsertKeyFrame(Frame::Ptr frame);

        void InsertMapPoint(MapPoint::Ptr map_point);

        LandmarksType GetAllMapPoints() {
            unique_lock<mutex> lck(data_mutex_);
            return landmarks_;
        }

        KeyframesType GetAllKeyFrames() {
            unique_lock<mutex> lck(data_mutex_);
            return keyframes_;
        }

        LandmarksType GetActiveMapPoints() {
            unique_lock<mutex> lck(data_mutex_);
            return active_landmarks_;
        }

        KeyframesType GetActiveKeyFrames() {
            unique_lock<mutex> lck(data_mutex_);
            return active_keyframes_;
        }

        void CleanMap();

    private:

        void RemoveOldKeyframe();

        mutex data_mutex_;
        LandmarksType landmarks_;
        LandmarksType active_landmarks_;
        KeyframesType keyframes_;
        KeyframesType active_keyframes_;

        Frame::Ptr current_frame_ = nullptr;

        // settings
        int num_active_keyframes_ = 7;
    };
}

#endif //MYSLAM_MAP_H