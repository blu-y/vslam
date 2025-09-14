//
// Created by Blu on 25. 9. 11..
//

#pragma once
#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "mySLAM/backend.h"
#include "mySLAM/common_include.h"
#include "mySLAM/dataset.h"
#include "mySLAM/frontend.h"
#include "mySLAM/viewer.h"

namespace mySLAM {

/**
 * VO External interface
 */
    class VisualOdometry {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<VisualOdometry> Ptr;

        /// constructor with config file
        VisualOdometry(string &config_path);

        /**
         * do initialization before run
         * @return true if success
         */
        bool Init();

        /**
         * start VO in the dataset
         */
        void Run();

        /**
         * Make a step forward in dataset
         * @return true if success
         */
        bool Step();

        /// get frontend status
        FrontendStatus GetFrontendStatus() const { return frontend_->Get_Status(); }

    private:
        bool inited_ = false;
        string config_file_path_;

        Frontend::Ptr frontend_ = nullptr;
        Backend::Ptr backend_ = nullptr;
        Map::Ptr map_ = nullptr;
        Viewer::Ptr viewer_ = nullptr;

        // dataset
        Dataset::Ptr dataset_ = nullptr;
    };
}

#endif //MYSLAM_VISUAL_ODOMETRY_H