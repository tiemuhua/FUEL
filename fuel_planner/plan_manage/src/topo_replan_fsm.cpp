
#include <plan_manage/topo_replan_fsm.h>

namespace fast_planner {
    TopoReplanFSM::TopoReplanFSM(ros::NodeHandle &nh) {
        current_wp_ = 0;
        exec_state_ = FSM_EXEC_STATE::INIT;
        have_target_ = false;
        collide_ = false;

        /*  fsm param  */
        nh.param("fsm/flight_type", target_type_, -1);
        nh.param("fsm/thresh_replan", replan_time_threshold_, -1.0);
        nh.param("fsm/thresh_no_replan", replan_distance_threshold_, -1.0);
        nh.param("fsm/waypoint_num", waypoint_num_, -1);
        nh.param("fsm/act_map", act_map_, false);
        for (size_t i = 0; i < waypoint_num_; i++) {
            nh.param("fsm/waypoint" + to_string(i) + "_x", waypoints_[i][0], -1.0);
            nh.param("fsm/waypoint" + to_string(i) + "_y", waypoints_[i][1], -1.0);
            nh.param("fsm/waypoint" + to_string(i) + "_z", waypoints_[i][2], -1.0);
        }

        /* initialize main modules */
        planner_manager_.reset(new FastPlannerManager(nh));
        visualization_.reset(new PlanningVisualization(nh));

        /* callback */
        exec_timer_ = nh.createTimer(ros::Duration(0.01), &TopoReplanFSM::execFSMCallback, this);
        safety_timer_ = nh.createTimer(ros::Duration(0.05), &TopoReplanFSM::checkCollisionCallback, this);

        waypoint_sub_ =
                nh.subscribe("/waypoint_generator/waypoints", 1, &TopoReplanFSM::waypointCallback, this);
        odom_sub_ = nh.subscribe("/odom_world", 1, &TopoReplanFSM::odometryCallback, this);

        replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 20);
        new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 20);
        bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 20);
    }

    void TopoReplanFSM::waypointCallback(const nav_msgs::PathConstPtr &msg) {
        if (msg->poses[0].pose.position.z < -0.1) return;
        cout << "Triggered!" << endl;

        vector<Eigen::Vector3d> global_wp;
        if (target_type_ == TARGET_TYPE::REFENCE_PATH) {
            for (size_t i = 0; i < waypoint_num_; ++i) {
                Eigen::Vector3d pt;
                pt(0) = waypoints_[i][0];
                pt(1) = waypoints_[i][1];
                pt(2) = waypoints_[i][2];
                global_wp.push_back(pt);
            }
        } else {
            if (target_type_ == TARGET_TYPE::MANUAL_TARGET) {
                target_point_(0) = msg->poses[0].pose.position.x;
                target_point_(1) = msg->poses[0].pose.position.y;
                target_point_(2) = 1.0;
                std::cout << "manual: " << target_point_.transpose() << std::endl;
            } else if (target_type_ == TARGET_TYPE::PRESET_TARGET) {
                target_point_(0) = waypoints_[current_wp_][0];
                target_point_(1) = waypoints_[current_wp_][1];
                target_point_(2) = waypoints_[current_wp_][2];

                current_wp_ = (current_wp_ + 1) % waypoint_num_;
                std::cout << "preset: " << target_point_.transpose() << std::endl;
            }

            global_wp.push_back(target_point_);
            visualization_->drawGoal(target_point_, 0.3, Eigen::Vector4d(1, 0, 0, 1.0));
        }

        planner_manager_->setGlobalWaypoints(global_wp);
        end_vel_.setZero();
        have_target_ = true;
        trigger_ = true;

        if (exec_state_ == WAIT_TARGET) {
            changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
        }
    }

    void TopoReplanFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg) {
        odom_pos_(0) = msg->pose.pose.position.x;
        odom_pos_(1) = msg->pose.pose.position.y;
        odom_pos_(2) = msg->pose.pose.position.z;

        odom_vel_(0) = msg->twist.twist.linear.x;
        odom_vel_(1) = msg->twist.twist.linear.y;
        odom_vel_(2) = msg->twist.twist.linear.z;

        odom_orient_.w() = msg->pose.pose.orientation.w;
        odom_orient_.x() = msg->pose.pose.orientation.x;
        odom_orient_.y() = msg->pose.pose.orientation.y;
        odom_orient_.z() = msg->pose.pose.orientation.z;

        have_odom_ = true;
    }

    void TopoReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, const string& pos_call) {
        string state_str[7] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "REPLAN_","NEW"};
        int pre_s = int(exec_state_);
        exec_state_ = new_state;
        cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;
    }

    void TopoReplanFSM::printFSMExecState() {
        string state_str[7] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "REPLAN_","NEW"};
        cout << "state: " + state_str[int(exec_state_)] << endl;
    }

    void TopoReplanFSM::execFSMCallback(const ros::TimerEvent &e) {
        static int fsm_num = 0;
        fsm_num++;
        if (fsm_num == 100) {
            printFSMExecState();
            if (!have_odom_) cout << "no odom." << endl;
            if (!trigger_) cout << "no trigger_." << endl;
            fsm_num = 0;
        }

        switch (exec_state_) {
            case INIT: {
                if (!have_odom_) return;
                if (!trigger_) return;
                changeFSMExecState(WAIT_TARGET, "FSM");

                break;
            }

            case WAIT_TARGET: {
                if (!have_target_)
                    return;
                else
                    changeFSMExecState(GEN_NEW_TRAJ, "FSM");

                break;
            }

            case GEN_NEW_TRAJ: {
                start_pt_ = odom_pos_;
                start_vel_ = odom_vel_;
                start_acc_.setZero();

                Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
                start_yaw_(0) = atan2(rot_x(1), rot_x(0));
                start_yaw_(1) = start_yaw_(2) = 0.0;

                new_pub_.publish(std_msgs::Empty());
                /* topo path finding and optimization */
                bool success = callTopologicalTraj(1);
                if (success)
                    changeFSMExecState(EXEC_TRAJ, "FSM");
                else {
                    ROS_WARN("Replan: retrying============================================");
                    changeFSMExecState(GEN_NEW_TRAJ, "FSM");
                }
                break;
            }

            case EXEC_TRAJ: {
                /* determine if need to replan */

                GlobalTrajDataPtr global_data = planner_manager_->global_data_;
                ros::Time time_now = ros::Time::now();
                double t_cur = (time_now - global_data->global_start_time_).toSec();

                if (t_cur > global_data->global_duration_ - 1e-2) {
                    have_target_ = false;
                    changeFSMExecState(WAIT_TARGET, "FSM");
                } else {
                    LocalTrajDataPtr info = planner_manager_->local_data_;
                    t_cur = (time_now - info->start_time_).toSec();

                    if (t_cur > replan_time_threshold_) {
                        if (!global_data->localTrajReachTarget()) {
                            ROS_WARN("Replan: periodic call=======================================");
                            changeFSMExecState(REPLAN_TRAJ, "FSM");
                        } else {
                            Eigen::Vector3d cur_pos = info->position_traj_.evaluateDeBoorT(t_cur);
                            Eigen::Vector3d end_pos = info->position_traj_.evaluateDeBoorT(info->duration_);
                            if ((cur_pos - end_pos).norm() > replan_distance_threshold_) {
                                ROS_WARN("Replan: periodic call=======================================");
                                changeFSMExecState(REPLAN_TRAJ, "FSM");
                            }
                        }
                    }
                }
                break;
            }

            case REPLAN_TRAJ: {
                LocalTrajDataPtr info = planner_manager_->local_data_;
                ros::Time time_now = ros::Time::now();
                double t_cur = (time_now - info->start_time_).toSec();

                start_pt_ = info->position_traj_.evaluateDeBoorT(t_cur);
                start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
                start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);

                start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_cur)[0];
                start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_cur)[0];
                start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_cur)[0];

                bool success = callTopologicalTraj(2);
                if (success) {
                    changeFSMExecState(EXEC_TRAJ, "FSM");
                } else {
                    ROS_WARN("Replan fail, retrying...");
                }

                break;
            }
            case REPLAN_NEW: {
                LocalTrajDataPtr info = planner_manager_->local_data_;
                ros::Time time_now = ros::Time::now();
                double t_cur = (time_now - info->start_time_).toSec();

                start_pt_ = info->position_traj_.evaluateDeBoorT(t_cur);
                start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
                start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);

                /* inform server */
                new_pub_.publish(std_msgs::Empty());

                // bool success = callSearchAndOptimization();
                bool success = callTopologicalTraj(1);
                if (success) {
                    changeFSMExecState(EXEC_TRAJ, "FSM");
                } else {
                    changeFSMExecState(GEN_NEW_TRAJ, "FSM");
                }

                break;
            }
        }
    }

    void TopoReplanFSM::checkCollisionCallback(const ros::TimerEvent &e) {
        LocalTrajDataPtr info = planner_manager_->local_data_;

        /* ---------- check goal safety ---------- */

        /* ---------- check trajectory ---------- */
        if (exec_state_ == EXEC_TRAJ || exec_state_ == REPLAN_TRAJ) {
            double dist;
            bool safe = planner_manager_->checkTrajCollision(dist);
            if (!safe) {
                if (dist > 0.2) {
                    ROS_WARN("current traj %lf m to collision", dist);
                    ROS_WARN("Replan: collision detected==================================");
                    collide_ = true;
                    changeFSMExecState(REPLAN_TRAJ, "SAFETY");
                } else {
                    ROS_ERROR("current traj %lf m to collision", dist);
                    replan_pub_.publish(std_msgs::Empty());
                    have_target_ = false;
                    changeFSMExecState(WAIT_TARGET, "SAFETY");
                }
            } else {
                collide_ = false;
            }
        }
    }

    void TopoReplanFSM::frontierCallback(const ros::TimerEvent &e) {
        if (!have_odom_) return;
        planner_manager_->searchFrontier(odom_pos_);
        visualization_->drawFrontier(planner_manager_->plan_data_->frontiers_);
    }

    bool TopoReplanFSM::callSearchAndOptimization() {
    }

    bool TopoReplanFSM::callTopologicalTraj(int step) {
        bool plan_success;

        if (step == 1) plan_success = planner_manager_->planGlobalTraj(start_pt_);

        replan_time_.push_back(0.0);
        auto t1 = ros::Time::now();
        plan_success = planner_manager_->topoReplan(collide_);
        replan_time_[replan_time_.size() - 1] += (ros::Time::now() - t1).toSec();

        if (plan_success) {
            if (!act_map_) {
                planner_manager_->planYaw(start_yaw_);
            } else {
                replan_time2_.push_back(0);
                auto t1 = ros::Time::now();
                planner_manager_->planYawActMap(start_yaw_);
                replan_time2_[replan_time2_.size() - 1] += (ros::Time::now() - t1).toSec();
            }

            LocalTrajDataPtr local_data = planner_manager_->local_data_;

            /* publish newest trajectory to server */

            /* publish traj */
            bspline::Bspline bspline;
            bspline.order = planner_manager_->pp_.bspline_degree_;
            bspline.start_time = local_data->start_time_;
            bspline.traj_id = local_data->traj_id_;

            Eigen::MatrixXd pos_pts = local_data->position_traj_.getControlPoint();

            for (Eigen::Index i = 0; i < pos_pts.rows(); ++i) {
                geometry_msgs::Point pt;
                pt.x = pos_pts(i, 0);
                pt.y = pos_pts(i, 1);
                pt.z = pos_pts(i, 2);
                bspline.pos_pts.push_back(pt);
            }

            Eigen::VectorXd knots = local_data->position_traj_.getKnot();
            for (Eigen::Index i = 0; i < knots.rows(); ++i) {
                bspline.knots.push_back(knots(i));
            }

            Eigen::MatrixXd yaw_pts = local_data->yaw_traj_.getControlPoint();
            for (Eigen::Index i = 0; i < yaw_pts.rows(); ++i) {
                double yaw = yaw_pts(i, 0);
                bspline.yaw_pts.push_back(yaw);
            }
            bspline.yaw_dt = local_data->yaw_traj_.getKnotSpan();

            bspline_pub_.publish(bspline);

            /* visualize new trajectories */
            MidPlanDataPtr plan_data = planner_manager_->plan_data_;

            visualization_->drawPolynomialTraj(planner_manager_->global_data_->global_traj_, 0.05,
                                               Eigen::Vector4d(0, 0, 0, 1), 0);
            visualization_->drawBspline(local_data->position_traj_, 0.08, Eigen::Vector4d(1.0, 0.0, 0.0, 1), true,
                                        0.15, Eigen::Vector4d(1, 1, 0, 1), 99);
            visualization_->drawBsplinesPhase2(plan_data->topo_traj_pos1_, 0.08);
            visualization_->drawViewConstraint(plan_data->view_cons_);
            visualization_->drawYawTraj(local_data->position_traj_, local_data->yaw_traj_, plan_data->dt_yaw_);
            return true;
        } else {
            return false;
        }
    }
// TopoReplanFSM::
}  // namespace fast_planner
