//
// Created by Blu on 25. 9. 6..
//

#pragma once

#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "mySLAM/common_include.h"

namespace mySLAM {

    struct Frame;
    struct Feature;

    /**
     * Landmark class
     * Feature points are triangulated to form landmarks
     */
    struct MapPoint {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0;          // ID
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero();       // Position in world
        mutex data_mutex_;
        int observed_times_ = 0;        // being observed by feature matching algo.
        list<weak_ptr<Feature> > observations_;

        MapPoint() {}

        MapPoint(long id, Vec3 position);

        Vec3 Pos() {
            unique_lock<mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos(const Vec3 &pos) {
            unique_lock<mutex> lck(data_mutex_);
            pos_ = pos;
        }

        void AddObservation(shared_ptr<Feature> feature) {
            unique_lock<mutex> lck(data_mutex_);
            observations_.push_back(feature);
            observed_times_++;
        }

        void RemoveObservation(shared_ptr<Feature> feat);

        list<weak_ptr<Feature> > GetObs() {
            unique_lock<mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };

}


#endif //MYSLAM_MAPPOINT_H