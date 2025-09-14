//
// Created by Blu on 25. 9. 11..
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

#include <mySLAM/common_include.h>

namespace mySLAM {

    /**
     * linear triangulation with SVD
     * @param poses     poses
     * @param points    points in normalized plane
     * @param pt_world  triangulated point in the world
     * @return true if success
     */
    inline bool triangulation(const vector<SE3> &poses,
                              const vector<Vec3> points,
                              Vec3 &pt_world) {
        MatXX A(2 * poses.size(), 4);
        VecX b(2 * poses.size());
        b.setZero();
        for (size_t i = 0; i < poses.size(); ++i) {
            Mat34 m = poses[i].matrix3x4();
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
            return true; // poor solution, give up
        }
        return false;
    }

    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }
}

#endif //MYSLAM_ALGORITHM_H