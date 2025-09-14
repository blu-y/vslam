//
// Created by Blu on 25. 9. 10..
//

#include "mySLAM/config.h"

namespace mySLAM {
    bool Config::SetParameterFile(const string &filename) {
        if (config_ == nullptr)
            config_ = shared_ptr<Config>(new Config);
        config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
        if (config_->file_.isOpened() == false) {
            LOG(ERROR) << "parameter file " << filename << " does not exist.";
            config_->file_.release();
            return false;
        }
        return true;
    }

    Config::~Config() {
        if (file_.isOpened()) file_.release();
    }

    shared_ptr<Config> Config::config_ = nullptr;

}