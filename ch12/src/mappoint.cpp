//
// Created by Blu on 25. 9. 6..
//

#include "mySLAM/mappoint.h"
#include "mySLAM/feature.h"

namespace mySLAM {

    MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

    MapPoint::Ptr MapPoint::CreateNewMappoint() {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

    void MapPoint::RemoveObservation(shared_ptr<Feature> feat) {
        unique_lock<mutex> lck(data_mutex_);
        for (auto iter = observations_.begin(); iter != observations_.end(); iter++) {
            if (iter->lock() == feat) {
                observations_.erase(iter);
                feat->map_point_.reset();
                observed_times_--;
                break;
            }
        }
    }

}