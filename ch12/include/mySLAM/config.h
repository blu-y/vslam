//
// Created by Blu on 25. 9. 10..
//

#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "mySLAM/common_include.h"

namespace mySLAM {

    /**
     * Configuration class,
     * uses "SetParameterFile" to determine the configuration file
     * uses "Get" to retrieve the corresponding value
     * Singleton pattern
     */
    class Config {
    private:
        static shared_ptr<Config> config_;
        cv::FileStorage file_;

        Config() {} // private constructor makes a singleton
    public:
        ~Config() {} // close the file when deconstructing

        // set a new config file
        static bool SetParameterFile(const string &filename);

        // access the parameter values
        template <typename T>
        static T Get(const string &key) {
            return T(config_->file_[key]);
        }
    };

}

#endif //MYSLAM_CONFIG_H