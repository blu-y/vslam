//
// Created by Blu on 25. 9. 10..
//

#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "mySLAM/common_include.h"
#include "mySLAM/frame.h"
#include "mySLAM/map.h"

namespace mySLAM {

    class Viewer {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr;

        Viewer();

        void SetMap(Map::Ptr map) { map_ = map; }

        void Close();

        void AddCurrentFrame(Frame::Ptr current_frame);

        void UpdateMap();

    private:
        void ThreadLoop();

        void DrawFrame(Frame::Ptr frame, const float *color);

        void DrawMapPoints();

        void FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera);

        /// plot the features in current fframe into an image
        Mat PlotFrameImage();

        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        thread viewer_thread_;
        bool viewer_running_ = true;

        unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
        unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
        bool map_updated_ = false;

        mutex viewer_data_mutex_;
    };

}

#endif //MYSLAM_VIEWER_H