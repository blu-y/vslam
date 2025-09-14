//
// Created by Blu on 25. 9. 11..
//

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "mySLAM/common_include.h"
#include "mySLAM/frame.h"
#include "mySLAM/map.h"

namespace mySLAM {

    class Map;

    /**
     * Backend
     * Has a separate optimization thread that starts optimization
     * when the Map is updated (triggered by frontend)
     */
    class Backend {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef shared_ptr<Backend> Ptr;

        /// Starts backend thread in the constructor
        Backend();

        // Set cameras and fetch the params
        void SetCamera(Camera::Ptr left, Camera::Ptr right) {
            cam_left_ = left;
            cam_right_ = right;
        }

        void SetMap(shared_ptr<Map> map) { map_ = map; }

        /// triggers map update and start optimization
        void UpdateMap();

        /// stop the backend thread
        void Stop();

    private:

        void BackendLoop();

        /// Optimizes the given keyframes and landmarks
        void Optimize(Map::KeyframesType &keyframes, Map::LandmarksType & landmarks);

        shared_ptr<Map> map_;
        thread backend_thread_;
        mutex data_mutex_;

        condition_variable map_update_;
        atomic<bool> backend_running_;

        Camera::Ptr cam_left_ = nullptr;
        Camera::Ptr cam_right_ = nullptr;
    };

}

#endif //MYSLAM_BACKEND_H