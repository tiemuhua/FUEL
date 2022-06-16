#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <bspline/Bspline.h>

using std::vector;
using Eigen::Vector3d;

namespace fast_planner {
    struct FSMData {
        // FSM data
        bool trigger_, have_odom_, static_state_;
        vector<string> state_str_;

        Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
        Eigen::Quaterniond odom_orient_;
        double odom_yaw_;
    };

    struct FSMParam {
        double replan_thresh1_;
        double replan_thresh2_;
        double replan_thresh3_;
        double replan_time_;  // second
    };

    struct ExplorationData {
        vector<vector<Vector3d>> frontiers_;
        vector<vector<Vector3d>> dead_frontiers_;
        vector<pair<Vector3d, Vector3d>> frontier_boxes_;
        vector<Vector3d> points_;
        vector<Vector3d> averages_;
        vector<Vector3d> views_;
        vector<double> yaws_;
        vector<Vector3d> global_tour_;

        vector<Vector3d> refined_tour_;

        Vector3d next_goal_;
        vector<Vector3d> path_next_goal_;
    };
    typedef shared_ptr<ExplorationData> ExplorationDataPtr;

    struct ExplorationParam {
        // params
        bool refine_local_;
        int refined_num_;
        double refined_radius_;
        int top_view_num_;
        double max_decay_;
        string tsp_dir_;  // resource dir of tsp solver
        double relax_time_;
    };

}  // namespace fast_planner

#endif