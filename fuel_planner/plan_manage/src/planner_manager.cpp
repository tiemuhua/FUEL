// #include <fstream>
#include <plan_manage/planner_manager.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

#include <memory>
#include <thread>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <ostream>

namespace fast_planner {
// SECTION interfaces for setup and query

    FastPlannerManager::~FastPlannerManager() {
        std::cout << "des manager" << std::endl;
    }

    FastPlannerManager::FastPlannerManager(ros::NodeHandle &nh) {
        /* read algorithm parameters */

        nh.param("manager/max_vel", pp_.max_vel_, -1.0);
        nh.param("manager/max_acc", pp_.max_acc_, -1.0);
        nh.param("manager/max_jerk", pp_.max_jerk_, -1.0);
        nh.param("manager/accept_vel", pp_.accept_vel_, pp_.max_vel_ + 0.5);
        nh.param("manager/accept_acc", pp_.accept_acc_, pp_.max_acc_ + 0.5);
        nh.param("manager/max_yawdot", pp_.max_yawdot_, -1.0);
        nh.param("manager/dynamic_environment", pp_.dynamic_, -1);
        nh.param("manager/clearance_threshold", pp_.clearance_, -1.0);
        nh.param("manager/local_segment_length", pp_.local_traj_len_, -1.0);
        nh.param("manager/control_points_distance", pp_.ctrl_pt_dist, -1.0);
        nh.param("manager/bspline_degree", pp_.bspline_degree_, 3);
        nh.param("manager/min_time", pp_.min_time_, false);

        bool use_geometric_path, use_kinodynamic_path, use_topo_path, use_optimization,
                use_active_perception;
        nh.param("manager/use_geometric_path", use_geometric_path, false);
        nh.param("manager/use_kinodynamic_path", use_kinodynamic_path, false);
        nh.param("manager/use_topo_path", use_topo_path, false);
        nh.param("manager/use_optimization", use_optimization, false);
        nh.param("manager/use_active_perception", use_active_perception, false);

        local_data_ = make_shared<LocalTrajData>();
        local_data_->traj_id_ = 0;
        global_data_ = make_shared<GlobalTrajData>();
        plan_data_ = make_shared<MidPlanData>();
        sdf_map_.reset(new SDFMap);
        sdf_map_->initMap(nh);
        edt_environment_.reset(new EDTEnvironment);
        edt_environment_->setMap(sdf_map_);

        if (use_geometric_path) {
            astar_path_finder_ = std::make_unique<Astar>(nh, edt_environment_);
        }

        if (use_kinodynamic_path) {
            kino_path_finder_ = std::make_unique<KinodynamicAstar>(nh, edt_environment_);
        }

        if (use_optimization) {
            bspline_optimizers_.resize(10);
            for (size_t i = 0; i < 10; ++i) {
                bspline_optimizers_[i] = std::make_unique<BsplineOptimizer>(nh, edt_environment_);
            }
        }

        if (use_active_perception) {
            frontier_finder_ = std::make_unique<FrontierFinder>(edt_environment_, nh);
            heading_planner_ = std::make_unique<HeadingPlanner>(nh, sdf_map_);
            visib_util_ = std::make_unique<VisibilityUtil>(nh);
            visib_util_->setEDTEnvironment(edt_environment_);
            plan_data_->view_cons_.idx_ = -1;
        }
    }

    void FastPlannerManager::setGlobalWaypoints(vector<Eigen::Vector3d> &waypoints) {
        plan_data_->global_waypoints_ = waypoints;
    }

    bool FastPlannerManager::checkTrajCollision(double &distance) {
        double t_now = (ros::Time::now() - local_data_->start_time_).toSec();

        Eigen::Vector3d cur_pt = local_data_->position_traj_.evaluateDeBoorT(t_now);
        double radius = 0.0;
        Eigen::Vector3d fut_pt;
        double fut_t = 0.02;

        while (radius < 6.0 && t_now + fut_t < local_data_->duration_) {
            fut_pt = local_data_->position_traj_.evaluateDeBoorT(t_now + fut_t);
            // double dist = edt_environment_->sdf_map_->getDistance(fut_pt);
            if (sdf_map_->getInflateOccupancy(fut_pt) == 1) {
                distance = radius;
                // std::cout << "collision at: " << fut_pt.transpose() << ", dist: " << dist << std::endl;
                std::cout << "collision at: " << fut_pt.transpose() << std::endl;
                return false;
            }
            radius = (fut_pt - cur_pt).norm();
            fut_t += 0.02;
        }

        return true;
    }

// !SECTION

// SECTION kinodynamic replanning

    bool FastPlannerManager::kinodynamicReplan(const Eigen::Vector3d &start_pt,
                                               const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                               const Eigen::Vector3d &end_pt, const Eigen::Vector3d &end_vel,
                                               const double &time_lb) {
        std::cout << "[Kino replan]: start: " << start_pt.transpose() << ", " << start_vel.transpose()
                  << ", " << start_acc.transpose() << ", goal:" << end_pt.transpose() << ", "
                  << end_vel.transpose() << endl;

        if ((start_pt - end_pt).norm() < 1e-2) {
            cout << "Close goal" << endl;
            return false;
        }

        /******************************
         * Kinodynamic path searching *
         ******************************/
        vector<PathNodePtr> path;
        double shot_time;
        Eigen::MatrixXd coef_shot;
        bool is_shot_succ;

        const bool dynamic = false;
        const double time_start = -1.0;

        int status = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, dynamic, time_start,
                                               true, path, is_shot_succ, coef_shot, shot_time);
        if (status == KinodynamicAstar::NO_PATH) {
            ROS_ERROR("search 1 fail");
            status = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, dynamic, time_start,
                                               false, path, is_shot_succ, coef_shot, shot_time);
            if (status == KinodynamicAstar::NO_PATH) {
                cout << "[Kino replan]: Can't find path." << endl;
                return false;
            }
        }
        plan_data_->kino_path_ = KinodynamicAstar::getKinoTraj(path, is_shot_succ, coef_shot, shot_time);

        /*********************************
         * Parameterize path to B-spline *
         *********************************/
        double ts = pp_.ctrl_pt_dist / pp_.max_vel_;
        vector<Eigen::Vector3d> point_set, start_end_derivatives;
        KinodynamicAstar::getSamples(path, start_vel, end_vel, is_shot_succ, coef_shot, shot_time, ts, point_set, start_end_derivatives);
        Eigen::MatrixXd ctrl_pts;
        NonUniformBspline::parameterizeToBspline(
                ts, point_set, start_end_derivatives, pp_.bspline_degree_, ctrl_pts);
        NonUniformBspline init_bspline(ctrl_pts, pp_.bspline_degree_, ts);

        /*********************************
         * B-spline-based optimization   *
         *********************************/
        int cost_function = BsplineOptimizer::NORMAL_PHASE;
        if (pp_.min_time_) cost_function |= BsplineOptimizer::MINTIME;
        vector<Eigen::Vector3d> start, end;
        init_bspline.getBoundaryStates(2, 0, start, end);
        bspline_optimizers_[0]->setBoundaryStates(start, end);
        if (time_lb > 0) bspline_optimizers_[0]->setTimeLowerBound(time_lb);
        bspline_optimizers_[0]->optimize(ctrl_pts, ts, cost_function, 1, 1);
        local_data_->position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, ts);

        vector<Eigen::Vector3d> start2, end2;
        local_data_->position_traj_.getBoundaryStates(2, 0, start2, end2);
        std::cout << "State error: (" << (start2[0] - start[0]).norm() << ", "
                  << (start2[1] - start[1]).norm() << ", " << (start2[2] - start[2]).norm() << ")"
                  << std::endl;

        updateTrajInfo();
        return true;
    }

    void FastPlannerManager::updateTrajInfo() {
        local_data_->velocity_traj_ = local_data_->position_traj_.getDerivative();
        local_data_->acceleration_traj_ = local_data_->velocity_traj_.getDerivative();

        local_data_->start_pos_ = local_data_->position_traj_.evaluateDeBoorT(0.0);
        local_data_->duration_ = local_data_->position_traj_.getTimeSum();

        local_data_->traj_id_ += 1;
    }

    void FastPlannerManager::planExploreTraj(const vector<Eigen::Vector3d> &tour,
                                             const Eigen::Vector3d &cur_vel, const Eigen::Vector3d &cur_acc,
                                             const double &time_lb) {
        if (tour.empty()) ROS_ERROR("Empty path to traj planner");

        // Generate traj through waypoints-based method
        const size_t pt_num = tour.size();
        Eigen::MatrixXd pos(pt_num, 3);
        for (Eigen::Index i = 0; i < pt_num; ++i) pos.row(i) = tour[i];

        Eigen::Vector3d zero(0, 0, 0);
        Eigen::VectorXd times(pt_num - 1);
        for (Eigen::Index i = 0; i < pt_num - 1; ++i)
            times(i) = (pos.row(i + 1) - pos.row(i)).norm() / (pp_.max_vel_ * 0.5);

        PolynomialTraj init_traj;
        PolynomialTraj::waypointsTraj(pos, cur_vel, zero, cur_acc, zero, times, init_traj);

        // B-spline-based optimization
        vector<Vector3d> points, boundary_deri;
        double duration = init_traj.getTotalTime();
        int seg_num = init_traj.getLength() / pp_.ctrl_pt_dist;
        seg_num = max(8, seg_num);
        double dt = duration / double(seg_num);

        std::cout << "duration: " << duration << ", seg_num: " << seg_num << ", dt: " << dt << std::endl;

        for (double ts = 0.0; ts <= duration + 1e-4; ts += dt)
            points.push_back(init_traj.evaluate(ts, 0));
        boundary_deri.push_back(init_traj.evaluate(0.0, 1));
        boundary_deri.push_back(init_traj.evaluate(duration, 1));
        boundary_deri.push_back(init_traj.evaluate(0.0, 2));
        boundary_deri.push_back(init_traj.evaluate(duration, 2));

        Eigen::MatrixXd ctrl_pts;
        NonUniformBspline::parameterizeToBspline(
                dt, points, boundary_deri, pp_.bspline_degree_, ctrl_pts);
        NonUniformBspline tmp_traj(ctrl_pts, pp_.bspline_degree_, dt);

        int cost_func = BsplineOptimizer::NORMAL_PHASE;
        if (pp_.min_time_) cost_func |= BsplineOptimizer::MINTIME;

        vector<Vector3d> start, end;
        tmp_traj.getBoundaryStates(2, 0, start, end);
        bspline_optimizers_[0]->setBoundaryStates(start, end);
        if (time_lb > 0) bspline_optimizers_[0]->setTimeLowerBound(time_lb);

        bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_func, 1, 1);
        local_data_->position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

        updateTrajInfo();
    }

// !SECTION

    void FastPlannerManager::planYaw(const Eigen::Vector3d &start_yaw) {
        auto t1 = ros::Time::now();
        // calculate waypoints of heading

        auto &pos = local_data_->position_traj_;
        double duration = pos.getTimeSum();

        double dt_yaw = 0.3;
        int seg_num = ceil(duration / dt_yaw);
        dt_yaw = duration / seg_num;

        const double forward_t = 2.0;
        double last_yaw = start_yaw(0);
        vector<Eigen::Vector3d> waypts;
        vector<int> waypt_idx;

        // seg_num -> seg_num - 1 points for constraint excluding the boundary states

        for (int i = 0; i < seg_num; ++i) {
            double tc = i * dt_yaw;
            Eigen::Vector3d pc = pos.evaluateDeBoorT(tc);
            double tf = min(duration, tc + forward_t);
            Eigen::Vector3d pf = pos.evaluateDeBoorT(tf);
            Eigen::Vector3d pd = pf - pc;

            Eigen::Vector3d waypt;
            if (pd.norm() > 1e-6) {
                waypt(0) = atan2(pd(1), pd(0));
                waypt(1) = waypt(2) = 0.0;
                calcNextYaw(last_yaw, waypt(0));
            } else {
                waypt = waypts.back();
            }
            last_yaw = waypt(0);
            waypts.push_back(waypt);
            waypt_idx.push_back(i);
        }

        // calculate initial control points with boundary state constraints

        Eigen::MatrixXd yaw(seg_num + 3, 1);
        yaw.setZero();

        Eigen::Matrix3d states2pts;
        states2pts << 1.0, -dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw, 1.0, 0.0, -(1 / 6.0) * dt_yaw * dt_yaw,
                1.0, dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw;
        yaw.block(0, 0, 3, 1) = states2pts * start_yaw;

        Eigen::Vector3d end_v = local_data_->velocity_traj_.evaluateDeBoorT(duration - 0.1);
        Eigen::Vector3d end_yaw(atan2(end_v(1), end_v(0)), 0, 0);
        calcNextYaw(last_yaw, end_yaw(0));
        yaw.block(seg_num, 0, 3, 1) = states2pts * end_yaw;

        // solve
        bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
        int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::WAYPOINTS |
                        BsplineOptimizer::START | BsplineOptimizer::END;

        vector<Eigen::Vector3d> start = {Eigen::Vector3d(start_yaw[0], 0, 0),
                                         Eigen::Vector3d(start_yaw[1], 0, 0), Eigen::Vector3d(start_yaw[2], 0, 0)};
        vector<Eigen::Vector3d> end = {Eigen::Vector3d(end_yaw[0], 0, 0),
                                       Eigen::Vector3d(end_yaw[1], 0, 0), Eigen::Vector3d(end_yaw[2], 0, 0)};
        bspline_optimizers_[1]->setBoundaryStates(start, end);
        bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);

        // update traj info
        local_data_->yaw_traj_.setUniformBspline(yaw, pp_.bspline_degree_, dt_yaw);
        local_data_->yawdot_traj_ = local_data_->yaw_traj_.getDerivative();
        local_data_->yawdotdot_traj_ = local_data_->yawdot_traj_.getDerivative();

        vector<double> path_yaw;
        for (const Vector3d & waypt : waypts) path_yaw.push_back(waypt[0]);
        plan_data_->path_yaw_ = path_yaw;
        plan_data_->dt_yaw_ = dt_yaw;
        plan_data_->dt_yaw_path_ = dt_yaw;

        std::cout << "yaw time: " << (ros::Time::now() - t1).toSec() << std::endl;
    }

    void FastPlannerManager::planYawExplore(const Eigen::Vector3d &start_yaw, const double &end_yaw,
                                            bool lookfwd, const double &relax_time) {
        const int seg_num = 12;
        double dt_yaw = local_data_->duration_ / seg_num;  // time of B-spline segment
        Eigen::Vector3d start_yaw3d = start_yaw;
        std::cout << "dt_yaw: " << dt_yaw << ", start yaw: " << start_yaw3d.transpose()
                  << ", end: " << end_yaw << std::endl;

        while (start_yaw3d[0] < -M_PI) start_yaw3d[0] += 2 * M_PI;
        while (start_yaw3d[0] > M_PI) start_yaw3d[0] -= 2 * M_PI;
        double last_yaw = start_yaw3d[0];

        // Yaw traj control points
        Eigen::MatrixXd yaw(seg_num + 3, 1);
        yaw.setZero();

        // Initial state
        Eigen::Matrix3d states2pts;
        states2pts << 1.0, -dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw, 1.0, 0.0, -(1 / 6.0) * dt_yaw * dt_yaw,
                1.0, dt_yaw, (1 / 3.0) * dt_yaw * dt_yaw;
        yaw.block<3, 1>(0, 0) = states2pts * start_yaw3d;
        string log_file_name = "/home/gjt/fuel_ws/planner_manager.log";
        ofstream fout;
        fout.open(log_file_name);

        fout<<1<<endl;
        // Add waypoint constraints if look forward is enabled
        vector<Eigen::Vector3d> waypts;
        vector<int> waypt_idx;
        if (lookfwd) {
            const double forward_t = 2.0;
            const int relax_num = relax_time / dt_yaw;
            for (int i = 1; i < seg_num - relax_num; ++i) {
                double tc = i * dt_yaw;
                Eigen::Vector3d pc = local_data_->position_traj_.evaluateDeBoorT(tc);
                double tf = min(local_data_->duration_, tc + forward_t);
                Eigen::Vector3d pf = local_data_->position_traj_.evaluateDeBoorT(tf);
                Eigen::Vector3d pd = pf - pc;
                Eigen::Vector3d waypt;
                if (pd.norm() > 1e-6) {
                    waypt(0) = atan2(pd(1), pd(0));
                    waypt(1) = waypt(2) = 0.0;
                    calcNextYaw(last_yaw, waypt(0));
                } else
                    waypt = waypts.back();

                last_yaw = waypt(0);
                waypts.push_back(waypt);
                waypt_idx.push_back(i);
            }
        }
        fout<<2<<endl;

        // Final state
        Eigen::Vector3d end_yaw3d(end_yaw, 0, 0);
        calcNextYaw(last_yaw, end_yaw3d(0));
        yaw.block<3, 1>(seg_num, 0) = states2pts * end_yaw3d;
        fout<<3<<endl;

        // Debug rapid change of yaw
        if (fabs(start_yaw3d[0] - end_yaw3d[0]) >= M_PI) {
            ROS_ERROR("Yaw change rapidly!");
            std::cout << "start yaw: " << start_yaw3d[0] << ", " << end_yaw3d[0] << std::endl;
        }
        fout<<4<<endl;

        auto t1 = ros::Time::now();

        // Call B-spline optimization solver
        int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::START | BsplineOptimizer::END |
                        BsplineOptimizer::WAYPOINTS;
        vector<Eigen::Vector3d> start = {Eigen::Vector3d(start_yaw3d[0], 0, 0),
                                         Eigen::Vector3d(start_yaw3d[1], 0, 0), Eigen::Vector3d(start_yaw3d[2], 0, 0)};
        vector<Eigen::Vector3d> end = {Eigen::Vector3d(end_yaw3d[0], 0, 0), Eigen::Vector3d(0, 0, 0)};
        fout<<5<<endl;
        bspline_optimizers_[1]->setBoundaryStates(start, end);
        fout<<6<<endl;
        bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
        fout<<7<<endl;
        bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);
        fout<<8<<endl;

        // Update traj info
        local_data_->yaw_traj_.setUniformBspline(yaw, 3, dt_yaw);
        fout<<9<<endl;
        local_data_->yawdot_traj_ = local_data_->yaw_traj_.getDerivative();
        fout<<10<<endl;
        local_data_->yawdotdot_traj_ = local_data_->yawdot_traj_.getDerivative();
        fout<<11<<endl;
        plan_data_->dt_yaw_ = dt_yaw;
    }

    void FastPlannerManager::calcNextYaw(const double &last_yaw, double &yaw) {
        // round yaw to [-PI, PI]
        double round_last = last_yaw;
        while (round_last < -M_PI) {
            round_last += 2 * M_PI;
        }
        while (round_last > M_PI) {
            round_last -= 2 * M_PI;
        }

        double diff = yaw - round_last;
        if (fabs(diff) <= M_PI) {
            yaw = last_yaw + diff;
        } else if (diff > M_PI) {
            yaw = last_yaw + diff - 2 * M_PI;
        } else if (diff < -M_PI) {
            yaw = last_yaw + diff + 2 * M_PI;
        }
    }

}  // namespace fast_planner
