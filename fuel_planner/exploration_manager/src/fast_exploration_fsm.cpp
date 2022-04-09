
#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

#include <exploration_manager/fast_exploration_fsm.h>
#include <exploration_manager/expl_data.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>

using Eigen::Vector4d;

namespace fast_planner {
//    void FastExplorationFSM::init(ros::NodeHandle &nh) {
    FastExplorationFSM::FastExplorationFSM(ros::NodeHandle &nh) {
        fp_.reset(new FSMParam);
        fd_.reset(new FSMData);

        /*  Fsm param  */
        nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
        nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
        nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
        nh.param("fsm/replan_time", fp_->replan_time_, -1.0);

        /* Initialize main modules */
        expl_manager_.reset(new FastExplorationManager(nh));
        visualization_.reset(new PlanningVisualization(nh));

        planner_manager_ = expl_manager_->planner_manager_;
        state_ = EXPL_STATE::INIT;
        fd_->have_odom_ = false;
        fd_->state_str_ = {"INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH"};
        fd_->static_state_ = true;
        fd_->trigger_ = false;

        /* Ros sub, pub and timer */
        exec_timer_ = nh.createTimer(ros::Duration(0.01), &FastExplorationFSM::FSMCallback, this);
        safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
        frontier_timer_ = nh.createTimer(ros::Duration(0.5), &FastExplorationFSM::frontierCallback, this);

        trigger_sub_ =
                nh.subscribe("/waypoint_generator/waypoints", 1, &FastExplorationFSM::triggerCallback, this);
        odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);

        replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
        new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
        bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);
    }

    void FastExplorationFSM::FSMCallback(const ros::TimerEvent &e) {
        ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << fd_->state_str_[int(state_)]);

        switch (state_) {
            case INIT: {
                // Wait for odometry ready
                if (!fd_->have_odom_) {
                    ROS_WARN_THROTTLE(1.0, "no odom.");
                    return;
                }
                // Go to wait trigger when odom is ok
                transitState(WAIT_TRIGGER, "FSM");
                break;
            }

            case WAIT_TRIGGER: {
                // Do nothing but wait for trigger
                ROS_WARN_THROTTLE(1.0, "wait for trigger.");
                break;
            }

            case FINISH: {
                ROS_INFO_THROTTLE(1.0, "finish exploration.");
                break;
            }

            case PLAN_TRAJ: {
                if (fd_->static_state_) {
                    // Plan from static state (hover)
                    fd_->start_pt_ = fd_->odom_pos_;
                    fd_->start_vel_ = fd_->odom_vel_;
                    fd_->start_acc_.setZero();

                    fd_->start_yaw_(0) = fd_->odom_yaw_;
                    fd_->start_yaw_(1) = fd_->start_yaw_(2) = 0.0;
                } else {
                    // Replan from non-static state, starting from 'replan_time' seconds later
                    LocalTrajDataPtr info = planner_manager_->local_data_;
                    double t_r = (ros::Time::now() - info->start_time_).toSec() + fp_->replan_time_;

                    fd_->start_pt_ = info->position_traj_.evaluateDeBoorT(t_r);
                    fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
                    fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
                    fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
                    fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
                    fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
                }

                // Inform traj_server the replanning
                replan_pub_.publish(std_msgs::Empty());
                int res = callExplorationPlanner();
                if (res == SUCCEED) {
                    transitState(PUB_TRAJ, "FSM");
                } else if (res == NO_FRONTIER) {
                    transitState(FINISH, "FSM");
                    fd_->static_state_ = true;
                } else if (res == FAIL) {
                    // Still in PLAN_TRAJ state, keep replanning
                    ROS_WARN("plan fail");
                    fd_->static_state_ = true;
                }
                break;
            }

            case PUB_TRAJ: {
                double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
                if (dt > 0) {
                    bspline_pub_.publish(fd_->newest_traj_);
                    fd_->static_state_ = false;
                    transitState(EXEC_TRAJ, "FSM");

                    thread vis_thread(&FastExplorationFSM::visualize, this);
                    vis_thread.detach();
                }
                break;
            }

            case EXEC_TRAJ: {
                LocalTrajDataPtr info = planner_manager_->local_data_;
                double t_cur = (ros::Time::now() - info->start_time_).toSec();

                // Replan if traj is almost fully executed
                double time_to_end = info->duration_ - t_cur;
                if (time_to_end < fp_->replan_thresh1_) {
                    transitState(PLAN_TRAJ, "FSM");
                    ROS_WARN("Replan: traj fully executed=================================");
                    return;
                }
                // Replan if next frontier to be visited is covered
                if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) {
                    transitState(PLAN_TRAJ, "FSM");
                    ROS_WARN("Replan: cluster covered=====================================");
                    return;
                }
                // Replan after some time
                if (t_cur > fp_->replan_thresh3_ && !classic_) {
                    transitState(PLAN_TRAJ, "FSM");
                    ROS_WARN("Replan: periodic call=======================================");
                }
                break;
            }
        }
    }

    int FastExplorationFSM::callExplorationPlanner() {
        ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);

        int res = expl_manager_->planExploreMotion(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_,
                                                   fd_->start_yaw_);
        classic_ = false;

        if (res == SUCCEED) {
            LocalTrajDataPtr info = planner_manager_->local_data_;
            info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;

            bspline::Bspline bspline;
            bspline.order = planner_manager_->pp_.bspline_degree_;
            bspline.start_time = info->start_time_;
            bspline.traj_id = info->traj_id_;
            Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
            for (Eigen::Index i = 0; i < pos_pts.rows(); ++i) {
                geometry_msgs::Point pt;
                pt.x = pos_pts(i, 0);
                pt.y = pos_pts(i, 1);
                pt.z = pos_pts(i, 2);
                bspline.pos_pts.push_back(pt);
            }
            Eigen::VectorXd knots = info->position_traj_.getKnot();
            for (Eigen::Index i = 0; i < knots.rows(); ++i) {
                bspline.knots.push_back(knots(i));
            }
            Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
            for (Eigen::Index i = 0; i < yaw_pts.rows(); ++i) {
                double yaw = yaw_pts(i, 0);
                bspline.yaw_pts.push_back(yaw);
            }
            bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
            fd_->newest_traj_ = bspline;
        }
        return res;
    }

    void FastExplorationFSM::visualize() {
        LocalTrajDataPtr info = planner_manager_->local_data_;
        ExplorationDataPtr ed_ptr = expl_manager_->ed_;

        // Draw frontier
        static size_t last_ftr_num = 0;
        for (size_t i = 0; i < ed_ptr->frontiers_.size(); ++i) {
            visualization_->drawCubes(ed_ptr->frontiers_[i], 0.1,
                                      visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.4),
                                      "frontier", i, 4);
        }
        // Search new frontier within box slightly inflated from updated box
        for (size_t i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
            visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
        }
        last_ftr_num = ed_ptr->frontiers_.size();
        visualization_->drawBspline(info->position_traj_, 0.1, Vector4d(1.0, 0.0, 0.0, 1), false, 0.15,
                                    Vector4d(1, 1, 0, 1));
    }

    void FastExplorationFSM::frontierCallback(const ros::TimerEvent &e) {
        static int delay = 0;
        if (++delay < 5) return;

        if (state_ == WAIT_TRIGGER || state_ == FINISH) {
            auto ft = expl_manager_->frontier_finder_;
            auto ed = expl_manager_->ed_;
            ft->removeOutDatedFrontiers();
            ft->searchAndAddFrontiers();
            ft->updateFrontierCostMatrix();

            ft->getFrontiers(ed->frontiers_);
            ft->getFrontierBoxes(ed->frontier_boxes_);

            // Draw frontier and bounding box
            for (size_t i = 0; i < ed->frontiers_.size(); ++i) {
                visualization_->drawCubes(ed->frontiers_[i], 0.1,
                                          visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4),
                                          "frontier", i, 4);
            }
            for (int i = ed->frontiers_.size(); i < 50; ++i) {
                visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
            }
        }
    }

    void FastExplorationFSM::triggerCallback(const nav_msgs::PathConstPtr &msg) {
        if (msg->poses[0].pose.position.z < -0.1) return;
        if (state_ != WAIT_TRIGGER) return;
        fd_->trigger_ = true;
        cout << "Triggered!" << endl;
        transitState(PLAN_TRAJ, "triggerCallback");
    }

    void FastExplorationFSM::safetyCallback(const ros::TimerEvent &e) {
        if (state_ == EXPL_STATE::EXEC_TRAJ) {
            // Check safety and trigger replan if necessary
            double dist;
            bool safe = planner_manager_->checkTrajCollision(dist);
            if (!safe) {
                ROS_WARN("Replan: collision detected==================================");
                transitState(PLAN_TRAJ, "safetyCallback");
            }
        }
    }

    void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg) {
        fd_->odom_pos_(0) = msg->pose.pose.position.x;
        fd_->odom_pos_(1) = msg->pose.pose.position.y;
        fd_->odom_pos_(2) = msg->pose.pose.position.z;

        fd_->odom_vel_(0) = msg->twist.twist.linear.x;
        fd_->odom_vel_(1) = msg->twist.twist.linear.y;
        fd_->odom_vel_(2) = msg->twist.twist.linear.z;

        fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
        fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
        fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
        fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

        Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
        fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

        fd_->have_odom_ = true;
    }

    void FastExplorationFSM::transitState(EXPL_STATE new_state, const string& pos_call) {
        int pre_s = int(state_);
        state_ = new_state;
        cout << "[" + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]
             << endl;
    }
}  // namespace fast_planner
