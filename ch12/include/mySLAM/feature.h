//
// Created by Blu on 25. 9. 6..
//

#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include "mySLAM/common_include.h"

namespace mySLAM {

    struct Frame;
    struct MapPoint;

    /**
     * 2D feature point
     * associated with map point after triangulation
     */

    struct Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<Feature> Ptr;

        weak_ptr<Frame> frame_;         // the frame holding this feature
        cv::KeyPoint position_;         // 2D pixel position
        weak_ptr<MapPoint> map_point_;  // assigned map point

        bool is_outlier_ = false;
        bool is_on_left_image_ = true;

        Feature() {}

        Feature(shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp) {}
    };

}


#endif //MYSLAM_FEATURE_H