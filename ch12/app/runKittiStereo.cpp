//
// Created by Blu on 25. 9. 6..
//

#include <gflags/gflags.h>
#include "mySLAM/visual_odometry.h"

using namespace mySLAM;

DEFINE_string(config_file, "../../config/default.yaml", "config file path");

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    VisualOdometry::Ptr VO(new VisualOdometry(FLAGS_config_file));
    assert(VO->Init() == true);
    VO->Run();

    return 0;
}