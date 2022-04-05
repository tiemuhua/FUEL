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

        local_data_.traj_id_ = 0;
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

        if (use_topo_path) {
            topo_prm_ = std::make_unique<TopologyPRM>(nh, edt_environment_);
        }

        if (use_active_perception) {
            frontier_finder_ = std::make_unique<FrontierFinder>(edt_environment_, nh);
            heading_planner_ = std::make_unique<HeadingPlanner>(nh, sdf_map_);
            visib_util_ = std::make_unique<VisibilityUtil>(nh);
            visib_util_->setEDTEnvironment(edt_environment_);
            plan_data_.view_cons_.idx_ = -1;
        }
    }

    void FastPlannerManager::setGlobalWaypoints(vector<Eigen::Vector3d> &waypoints) {
        plan_data_.global_waypoints_ = waypoints;
    }

    bool FastPlannerManager::checkTrajCollision(double &distance) {
        double t_now = (ros::Time::now() - local_data_.start_time_).toSec();

        Eigen::Vector3d cur_pt = local_data_.position_traj_.evaluateDeBoorT(t_now);
        double radius = 0.0;
        Eigen::Vector3d fut_pt;
        double fut_t = 0.02;

        while (radius < 6.0 && t_now + fut_t < local_data_.duration_) {
            fut_pt = local_data_.position_traj_.evaluateDeBoorT(t_now + fut_t);
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
        plan_data_.kino_path_ = KinodynamicAstar::getKinoTraj(path, is_shot_succ, coef_shot, shot_time);

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
        local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, ts);

        vector<Eigen::Vector3d> start2, end2;
        local_data_.position_traj_.getBoundaryStates(2, 0, start2, end2);
        std::cout << "State error: (" << (start2[0] - start[0]).norm() << ", "
                  << (start2[1] - start[1]).norm() << ", " << (start2[2] - start[2]).norm() << ")"
                  << std::endl;

        updateTrajInfo();
        return true;
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
        local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

        updateTrajInfo();
    }

// !SECTION

// SECTION topological replanning

    bool FastPlannerManager::planGlobalTraj(const Eigen::Vector3d &start_pos) {
        plan_data_.clearTopoPaths();

        // Generate global reference trajectory
        vector<Eigen::Vector3d> points = plan_data_.global_waypoints_;
        if (points.empty()) std::cout << "no global waypoints!" << std::endl;

        points.insert(points.begin(), start_pos);

        // Insert intermediate points if two waypoints are too far
        vector<Eigen::Vector3d> inter_points;
        const double dist_thresh = 4.0;

        for (size_t i = 0; i < points.size() - 1; ++i) {
            inter_points.push_back(points.at(i));
            double dist = (points.at(i + 1) - points.at(i)).norm();
            if (dist > dist_thresh) {
                int id_num = floor(dist / dist_thresh) + 1;
                for (int j = 1; j < id_num; ++j) {
                    Eigen::Vector3d inter_pt =
                            points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
                    inter_points.push_back(inter_pt);
                }
            }
        }
        inter_points.push_back(points.back());

        // At least 3 waypoints are required to solve the problem
        if (inter_points.size() == 2) {
            Eigen::Vector3d mid = (inter_points[0] + inter_points[1]) * 0.5;
            inter_points.insert(inter_points.begin() + 1, mid);
        }

        auto pt_num = (Eigen::Index)inter_points.size();
        Eigen::MatrixXd pos(pt_num, 3);
        for (Eigen::Index i = 0; i < pt_num; ++i) pos.row(i) = inter_points[i];

        Eigen::Vector3d zero(0, 0, 0);
        Eigen::VectorXd time(pt_num - 1);
        for (Eigen::Index i = 0; i < pt_num - 1; ++i)
            time(i) = (pos.row(i + 1) - pos.row(i)).norm() / (pp_.max_vel_ * 0.5);

        time(0) += pp_.max_vel_ / (2 * pp_.max_acc_);
        time(time.rows() - 1) += pp_.max_vel_ / (2 * pp_.max_acc_);

        PolynomialTraj gl_traj;
        PolynomialTraj::waypointsTraj(pos, zero, zero, zero, zero, time, gl_traj);

        auto time_now = ros::Time::now();
        global_data_.setGlobalTraj(gl_traj, time_now);

        // truncate a local trajectory

        double dt, duration;
        Eigen::MatrixXd ctrl_pts = paramLocalTraj(0.0, dt, duration);
        NonUniformBspline bspline(ctrl_pts, pp_.bspline_degree_, dt);

        std::cout << "ctrl pt: " << ctrl_pts.rows() << std::endl;

        global_data_.setLocalTraj(bspline, 0.0, duration, 0.0);
        local_data_.position_traj_ = bspline;
        local_data_.start_time_ = time_now;
        ROS_INFO("global trajectory generated.");

        updateTrajInfo();

        return true;
    }

    bool FastPlannerManager::topoReplan(bool collide) {
        ros::Time t1, t2;

        /* truncate a new local segment for replanning */
        ros::Time time_now = ros::Time::now();
        double t_now = (time_now - global_data_.global_start_time_).toSec();
        double local_traj_dt, local_traj_duration;

        Eigen::MatrixXd ctrl_pts = paramLocalTraj(t_now, local_traj_dt, local_traj_duration);
        NonUniformBspline init_traj(ctrl_pts, pp_.bspline_degree_, local_traj_dt);
        local_data_.start_time_ = time_now;

        std::cout << "dt: " << local_traj_dt << ", dur: " << local_traj_duration << std::endl;

        if (!collide) {
            // No collision detected, but we can further refine the trajectory
            refineTraj(init_traj);
            double time_change = init_traj.getTimeSum() - local_traj_duration;
            local_data_.position_traj_ = init_traj;
            global_data_.setLocalTraj(
                    local_data_.position_traj_, t_now, local_traj_duration + time_change + t_now, time_change);
            // local_data_.position_traj_ = init_traj;
            // global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0);
        } else {
            // Find topologically distinctive path and guide optimization in parallel
            plan_data_.initial_local_segment_ = init_traj;
            vector<Eigen::Vector3d> colli_start, colli_end, start_pts, end_pts;
            findCollisionRange(colli_start, colli_end, start_pts, end_pts);

            if (colli_start.size() == 1 && colli_end.empty()) {
                ROS_WARN("Init traj ends in obstacle, no replanning.");
                local_data_.position_traj_ = init_traj;
                global_data_.setLocalTraj(init_traj, t_now, local_traj_duration + t_now, 0.0);
            } else {
                // Call topological replanning when local segment is in collision
                /* Search topological distinctive paths */
                ROS_INFO("[Topo]: ---------");
                plan_data_.clearTopoPaths();
                list<GraphNode::Ptr> graph;
                vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths, select_paths;
                topo_prm_->findTopoPaths(colli_start.front(), colli_end.back(), start_pts, end_pts, graph,
                                         raw_paths, filtered_paths, select_paths);

                if (select_paths.empty()) {
                    ROS_WARN("No path.");
                    return false;
                }
                plan_data_.addTopoPaths(graph, raw_paths, filtered_paths, select_paths);

                /* Optimize trajectory using different topo guiding paths */
                ROS_INFO("[Optimize]: ---------");
                t1 = ros::Time::now();

                plan_data_.topo_traj_pos1_.resize(select_paths.size());
                plan_data_.topo_traj_pos2_.resize(select_paths.size());
                vector<thread> optimize_threads;
                for (size_t i = 0; i < select_paths.size(); ++i) {
                    optimize_threads.emplace_back(&FastPlannerManager::optimizeTopoBspline, this, t_now,
                                                  local_traj_duration, select_paths[i], i);
                }
                for (size_t i = 0; i < select_paths.size(); ++i) optimize_threads[i].join();

                double t_opt = (ros::Time::now() - t1).toSec();
                cout << "[planner]: optimization time: " << t_opt << endl;

                NonUniformBspline best_traj;
                selectBestTraj(best_traj);
                refineTraj(best_traj);
                double time_change = best_traj.getTimeSum() - local_traj_duration;

                local_data_.position_traj_ = best_traj;
                global_data_.setLocalTraj(local_data_.position_traj_, t_now,
                                          local_traj_duration + time_change + t_now, time_change);
            }
        }
        updateTrajInfo();

        double tr = (ros::Time::now() - time_now).toSec();
        ROS_WARN("Replan time: %lf", tr);

        return true;
    }

    void FastPlannerManager::selectBestTraj(NonUniformBspline &traj) {
        // sort by jerk
        vector<NonUniformBspline> &trajs = plan_data_.topo_traj_pos2_;
        sort(trajs.begin(), trajs.end(),
             [](NonUniformBspline &tj1, NonUniformBspline &tj2) { return tj1.getJerk() < tj2.getJerk(); });
        traj = trajs[0];
    }

    void FastPlannerManager::refineTraj(NonUniformBspline &best_traj) {
        ros::Time t1 = ros::Time::now();
        plan_data_.no_visib_traj_ = best_traj;

        int cost_function = BsplineOptimizer::NORMAL_PHASE;
        if (pp_.min_time_) cost_function |= BsplineOptimizer::MINTIME;

        // Refine selected best traj
        Eigen::MatrixXd ctrl_pts = best_traj.getControlPoint();
        double dt = best_traj.getKnotSpan();
        vector<Eigen::Vector3d> start1, end1;
        best_traj.getBoundaryStates(2, 0, start1, end1);

        bspline_optimizers_[0]->setBoundaryStates(start1, end1);
        bspline_optimizers_[0]->optimize(ctrl_pts, dt, cost_function, 2, 2);
        best_traj.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

        vector<Eigen::Vector3d> start2, end2;
        best_traj.getBoundaryStates(2, 2, start2, end2);
        for (size_t i = 0; i < 3; ++i)
            std::cout << "error start: " << (start1[i] - start2[i]).norm() << std::endl;
        for (size_t i = 0; i < 1; ++i)
            std::cout << "error end  : " << (end1[i] - end2[i]).norm() << std::endl;
    }

    void FastPlannerManager::updateTrajInfo() {
        local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
        local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();

        local_data_.start_pos_ = local_data_.position_traj_.evaluateDeBoorT(0.0);
        local_data_.duration_ = local_data_.position_traj_.getTimeSum();

        local_data_.traj_id_ += 1;
    }

    void FastPlannerManager::optimizeTopoBspline(
            double start_t, double duration, vector<Eigen::Vector3d> guide_path, int traj_id) {
        auto t1 = ros::Time::now();

        // Re-parameterize B-spline according to the length of guide path
        int seg_num = topo_prm_->pathLength(guide_path) / pp_.ctrl_pt_dist;
        seg_num = max(6, seg_num);  // Min number required for optimizing
        double dt = duration / double(seg_num);
        Eigen::MatrixXd ctrl_pts = reparamLocalTraj(start_t, duration, dt);

        NonUniformBspline tmp_traj(ctrl_pts, pp_.bspline_degree_, dt);
        vector<Eigen::Vector3d> start, end;
        tmp_traj.getBoundaryStates(2, 0, start, end);

        // std::cout << "ctrl pt num: " << ctrl_pts.rows() << std::endl;

        // Discretize the guide path and align it with B-spline control points
        vector<Eigen::Vector3d> tmp_pts, guide_pts;
        if (pp_.bspline_degree_ == 3 || pp_.bspline_degree_ == 5) {
            topo_prm_->pathToGuidePts(guide_path, int(ctrl_pts.rows()) - 2, tmp_pts);
            guide_pts.insert(guide_pts.end(), tmp_pts.begin() + 2, tmp_pts.end() - 2);
            if (guide_pts.size() != int(ctrl_pts.rows()) - 6) ROS_WARN("Incorrect guide for 3 degree");
        } else if (pp_.bspline_degree_ == 4) {
            topo_prm_->pathToGuidePts(guide_path, int(2 * ctrl_pts.rows()) - 7, tmp_pts);
            for (size_t i = 0; i < tmp_pts.size(); ++i) {
                if (i % 2 == 1 && i >= 5 && i <= tmp_pts.size() - 6) guide_pts.push_back(tmp_pts[i]);
            }
            if (guide_pts.size() != int(ctrl_pts.rows()) - 8) ROS_WARN("Incorrect guide for 4 degree");
        }

        // std::cout << "guide pt num: " << guide_pt.size() << std::endl;

        double tm1 = (ros::Time::now() - t1).toSec();
        t1 = ros::Time::now();

        // First phase, path-guided optimization
        bspline_optimizers_[traj_id]->setBoundaryStates(start, end);
        bspline_optimizers_[traj_id]->setGuidePath(guide_pts);
        bspline_optimizers_[traj_id]->optimize(ctrl_pts, dt, BsplineOptimizer::GUIDE_PHASE, 0, 1);
        plan_data_.topo_traj_pos1_[traj_id] = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

        double tm2 = (ros::Time::now() - t1).toSec();
        t1 = ros::Time::now();

        // Second phase, smooth+safety+feasibility
        int cost_func = BsplineOptimizer::NORMAL_PHASE;
        bspline_optimizers_[traj_id]->setBoundaryStates(start, end);
        bspline_optimizers_[traj_id]->optimize(ctrl_pts, dt, cost_func, 1, 1);
        plan_data_.topo_traj_pos2_[traj_id] = NonUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

        double tm3 = (ros::Time::now() - t1).toSec();
        // ROS_INFO("optimization %d cost %lf, %lf, %lf seconds.", traj_id, tm1, tm2, tm3);
    }

    Eigen::MatrixXd FastPlannerManager::paramLocalTraj(double start_t, double &dt, double &duration) {
        vector<Eigen::Vector3d> point_set;
        vector<Eigen::Vector3d> start_end_derivative;
        global_data_.getTrajInfoInSphere(start_t, pp_.local_traj_len_, pp_.ctrl_pt_dist, point_set,
                                         start_end_derivative, dt, duration);

        Eigen::MatrixXd ctrl_pts;
        NonUniformBspline::parameterizeToBspline(
                dt, point_set, start_end_derivative, pp_.bspline_degree_, ctrl_pts);
        plan_data_.local_start_end_derivative_ = start_end_derivative;

        return ctrl_pts;
    }

    Eigen::MatrixXd FastPlannerManager::reparamLocalTraj(
            const double &start_t, const double &duration, const double &dt) {
        vector<Eigen::Vector3d> point_set;
        vector<Eigen::Vector3d> start_end_derivative;

        global_data_.getTrajInfoInDuration(start_t, duration, dt, point_set, start_end_derivative);
        plan_data_.local_start_end_derivative_ = start_end_derivative;

        /* parameterization of B-spline */
        Eigen::MatrixXd ctrl_pts;
        NonUniformBspline::parameterizeToBspline(
                dt, point_set, start_end_derivative, pp_.bspline_degree_, ctrl_pts);
        // cout << "ctrl pts:" << ctrl_pts.rows() << endl;

        return ctrl_pts;
    }

    void FastPlannerManager::findCollisionRange(vector<Eigen::Vector3d> &colli_start,
                                                vector<Eigen::Vector3d> &colli_end, vector<Eigen::Vector3d> &start_pts,
                                                vector<Eigen::Vector3d> &end_pts) {
        bool last_safe = true, safe;
        double t_m, t_mp;
        NonUniformBspline *initial_traj = &plan_data_.initial_local_segment_;
        initial_traj->getTimeSpan(t_m, t_mp);

        /* find range of collision */
        double t_s = -1.0, t_e;
        for (double tc = t_m; tc <= t_mp + 1e-4; tc += 0.05) {
            Eigen::Vector3d ptc = initial_traj->evaluateDeBoor(tc);
            safe = edt_environment_->evaluateCoarseEDT(ptc, -1.0) >= topo_prm_->clearance_;

            if (last_safe && !safe) {
                colli_start.emplace_back(initial_traj->evaluateDeBoor(tc - 0.05));
                if (t_s < 0.0) t_s = tc - 0.05;
            } else if (!last_safe && safe) {
                colli_end.push_back(ptc);
                t_e = tc;
            }

            last_safe = safe;
        }

        if (colli_start.empty()) return;

        if (colli_start.size() == 1 && colli_end.empty()) return;

        /* find start and end safe segment */
        double dt = initial_traj->getKnotSpan();
        int sn = ceil((t_s - t_m) / dt);
        dt = (t_s - t_m) / sn;

        for (double tc = t_m; tc <= t_s + 1e-4; tc += dt) {
            start_pts.emplace_back(initial_traj->evaluateDeBoor(tc));
        }

        dt = initial_traj->getKnotSpan();
        sn = ceil((t_mp - t_e) / dt);
        dt = (t_mp - t_e) / sn;

        if (dt > 1e-4) {
            for (double tc = t_e; tc <= t_mp + 1e-4; tc += dt) {
                end_pts.emplace_back(initial_traj->evaluateDeBoor(tc));
            }
        } else {
            end_pts.emplace_back(initial_traj->evaluateDeBoor(t_mp));
        }
    }

// !SECTION

    void FastPlannerManager::planYaw(const Eigen::Vector3d &start_yaw) {
        auto t1 = ros::Time::now();
        // calculate waypoints of heading

        auto &pos = local_data_.position_traj_;
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

        Eigen::Vector3d end_v = local_data_.velocity_traj_.evaluateDeBoorT(duration - 0.1);
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
        local_data_.yaw_traj_.setUniformBspline(yaw, pp_.bspline_degree_, dt_yaw);
        local_data_.yawdot_traj_ = local_data_.yaw_traj_.getDerivative();
        local_data_.yawdotdot_traj_ = local_data_.yawdot_traj_.getDerivative();

        vector<double> path_yaw;
        for (size_t i = 0; i < waypts.size(); ++i) path_yaw.push_back(waypts[i][0]);
        plan_data_.path_yaw_ = path_yaw;
        plan_data_.dt_yaw_ = dt_yaw;
        plan_data_.dt_yaw_path_ = dt_yaw;

        std::cout << "yaw time: " << (ros::Time::now() - t1).toSec() << std::endl;
    }

    void FastPlannerManager::planYawExplore(const Eigen::Vector3d &start_yaw, const double &end_yaw,
                                            bool lookfwd, const double &relax_time) {
        const int seg_num = 12;
        double dt_yaw = local_data_.duration_ / seg_num;  // time of B-spline segment
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

        // Add waypoint constraints if look forward is enabled
        vector<Eigen::Vector3d> waypts;
        vector<int> waypt_idx;
        if (lookfwd) {
            const double forward_t = 2.0;
            const int relax_num = relax_time / dt_yaw;
            for (int i = 1; i < seg_num - relax_num; ++i) {
                double tc = i * dt_yaw;
                Eigen::Vector3d pc = local_data_.position_traj_.evaluateDeBoorT(tc);
                double tf = min(local_data_.duration_, tc + forward_t);
                Eigen::Vector3d pf = local_data_.position_traj_.evaluateDeBoorT(tf);
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
        // Final state
        Eigen::Vector3d end_yaw3d(end_yaw, 0, 0);
        calcNextYaw(last_yaw, end_yaw3d(0));
        yaw.block<3, 1>(seg_num, 0) = states2pts * end_yaw3d;

        // Debug rapid change of yaw
        if (fabs(start_yaw3d[0] - end_yaw3d[0]) >= M_PI) {
            ROS_ERROR("Yaw change rapidly!");
            std::cout << "start yaw: " << start_yaw3d[0] << ", " << end_yaw3d[0] << std::endl;
        }

        auto t1 = ros::Time::now();

        // Call B-spline optimization solver
        int cost_func = BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::START | BsplineOptimizer::END |
                        BsplineOptimizer::WAYPOINTS;
        vector<Eigen::Vector3d> start = {Eigen::Vector3d(start_yaw3d[0], 0, 0),
                                         Eigen::Vector3d(start_yaw3d[1], 0, 0), Eigen::Vector3d(start_yaw3d[2], 0, 0)};
        vector<Eigen::Vector3d> end = {Eigen::Vector3d(end_yaw3d[0], 0, 0), Eigen::Vector3d(0, 0, 0)};
        bspline_optimizers_[1]->setBoundaryStates(start, end);
        bspline_optimizers_[1]->setWaypoints(waypts, waypt_idx);
        bspline_optimizers_[1]->optimize(yaw, dt_yaw, cost_func, 1, 1);

        // std::cout << "2: " << (ros::Time::now() - t1).toSec() << std::endl;

        // Update traj info
        local_data_.yaw_traj_.setUniformBspline(yaw, 3, dt_yaw);
        local_data_.yawdot_traj_ = local_data_.yaw_traj_.getDerivative();
        local_data_.yawdotdot_traj_ = local_data_.yawdot_traj_.getDerivative();
        plan_data_.dt_yaw_ = dt_yaw;

        // plan_data_.path_yaw_ = path;
        // plan_data_.dt_yaw_path_ = dt_yaw * subsp;
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
