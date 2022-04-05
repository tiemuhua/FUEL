#include <path_searching/kinodynamic_astar.h>
#include <memory>
#include <sstream>
#include <plan_env/sdf_map.h>

#include <../../plan_manage/include/plan_manage/backward.hpp>

namespace backward {
    backward::SignalHandling sh;
}

using namespace std;
using namespace Eigen;

namespace fast_planner {
    KinodynamicAstar::KinodynamicAstar(ros::NodeHandle &nh, const EDTEnvironment::Ptr &env) {
        nh.param("search/max_tau", max_tau_, -1.0);
        nh.param("search/init_max_tau", init_max_tau_, -1.0);
        nh.param("search/max_vel", max_vel_, -1.0);
        nh.param("search/max_acc", max_acc_, -1.0);
        nh.param("search/w_time", w_time_, -1.0);
        nh.param("search/horizon", horizon_, -1.0);
        nh.param("search/resolution_astar", resolution_, -1.0);
        nh.param("search/time_resolution", time_resolution_, -1.0);
        nh.param("search/lambda_heu", lambda_heu_, -1.0);
        nh.param("search/allocate_num", allocate_num_, -1);
        nh.param("search/check_num", check_num_, -1);
        nh.param("search/optimistic", optimistic_, true);
        tie_breaker_ = 1.0 + 1.0 / 10000;

        double vel_margin;
        nh.param("search/vel_margin", vel_margin, 0.0);
        max_vel_ += vel_margin;

        /* ---------- map params ---------- */
        this->edt_environment_ = env;
        edt_environment_->sdf_map_->getRegion(origin_, map_size_3d_);

        cout << "kino origin_: " << origin_.transpose() << endl;
        cout << "kino map size: " << map_size_3d_.transpose() << endl;
    }

    int KinodynamicAstar::search(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &start_v,
                                 const Eigen::Vector3d &start_a,
                                 const Eigen::Vector3d &end_pt, const Eigen::Vector3d &end_v,
                                 const bool dynamic, const double time_start,
                                 bool init_search,
                                 vector<PathNodePtr> &path,
                                 bool &is_shot_succ, Eigen::MatrixXd &coef_shot, double &shot_time) {
        PathNodePtr cur_node = make_shared<PathNode>();
        cur_node->parent = nullptr;
        cur_node->state.head(3) = start_pt;
        cur_node->state.tail(3) = start_v;
        cur_node->index = posToIndex(start_pt);
        cur_node->g_score = 0.0;

        Eigen::VectorXd end_state(6);
        Eigen::Vector3i end_index;
        double time_to_goal;

        end_state.head(3) = end_pt;
        end_state.tail(3) = end_v;
        end_index = posToIndex(end_pt);
        cur_node->f_score = lambda_heu_ * estimateHeuristic(cur_node->state, end_state, time_to_goal);
        cur_node->node_state = IN_OPEN_SET;
        std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> open_set;
        open_set.push(cur_node);
        int use_node_num = 1;
        NodeHashTable expanded_nodes;

        if (dynamic) {
            time_origin_ = time_start;
            cur_node->time = time_start;
            cur_node->time_idx = timeToIndex(time_start);
            expanded_nodes.insert(cur_node->index, cur_node->time_idx, cur_node);
        } else
            expanded_nodes.insert(cur_node->index, cur_node);

        const int tolerance = ceil(1 / resolution_);

        int iter_num = 0;
        while (!open_set.empty()) {
            cur_node = open_set.top();
            open_set.pop();
            cur_node->node_state = IN_CLOSE_SET;
            iter_num += 1;

            // Terminate?
            bool reach_horizon = (cur_node->state.head(3) - start_pt).norm() >= horizon_;
            bool near_end = abs(cur_node->index(0) - end_index(0)) <= tolerance &&
                            abs(cur_node->index(1) - end_index(1)) <= tolerance &&
                            abs(cur_node->index(2) - end_index(2)) <= tolerance;
            if (reach_horizon || near_end) {
                path = move(retrievePath(cur_node));
                if (near_end) {
                    // Check whether shot traj exist
                    estimateHeuristic(cur_node->state, end_state, shot_time);
                    is_shot_succ = computeShotTraj(cur_node->state, end_state, shot_time, coef_shot);
                    if (init_search) ROS_ERROR("Shot in first search loop!");
                }
            }
            if (reach_horizon) {
                if (is_shot_succ) {
                    std::cout << "reach end" << std::endl;
                    return REACH_END;
                } else {
                    std::cout << "reach horizon" << std::endl;
                    return REACH_HORIZON;
                }
            }

            if (near_end) {
                if (is_shot_succ) {
                    std::cout << "reach end" << std::endl;
                    return REACH_END;
                } else if (cur_node->parent) {
                    std::cout << "near end" << std::endl;
                    return NEAR_END;
                } else {
                    std::cout << "no path" << std::endl;
                    return NO_PATH;
                }
            }

            double res = 1 / 2.0, time_res = 1 / 1.0, time_res_init = 1 / 20.0;
            Eigen::Matrix<double, 6, 1> cur_state = cur_node->state;
            Eigen::Matrix<double, 6, 1> pro_state;
            vector<PathNodePtr> tmp_expand_nodes;
            Eigen::Vector3d um;
            double pro_t;
            vector<Eigen::Vector3d> inputs;
            vector<double> durations;
            if (init_search) {
                inputs.push_back(start_a);
                for (double tau = time_res_init * init_max_tau_; tau <= init_max_tau_ + 1e-3;
                     tau += time_res_init * init_max_tau_)
                    durations.push_back(tau);
                init_search = false;
            } else {
                for (double ax = -max_acc_; ax <= max_acc_ + 1e-3; ax += max_acc_ * res)
                    for (double ay = -max_acc_; ay <= max_acc_ + 1e-3; ay += max_acc_ * res)
                        for (double az = -max_acc_; az <= max_acc_ + 1e-3; az += max_acc_ * res) {
                            um << ax, ay, az;
                            inputs.push_back(um);
                        }
                for (double tau = time_res * max_tau_; tau <= max_tau_; tau += time_res * max_tau_)
                    durations.push_back(tau);
            }

            // cout << "cur state:" << cur_state.head(3).transpose() << endl;
            for (const Vector3d &input: inputs) {
                for (double tau: durations) {
                    stateTransit(cur_state, pro_state, input, tau);
                    pro_t = cur_node->time + tau;

                    // Check inside map range
                    Eigen::Vector3d pro_pos = pro_state.head(3);
                    if (!edt_environment_->sdf_map_->isInBox(pro_pos)) {
                        if (init_search) std::cout << "box" << std::endl;
                        continue;
                    }

                    // Check if in close set
                    Eigen::Vector3i pro_id = posToIndex(pro_pos);
                    int pro_t_id = floor((pro_t - time_origin_) / time_resolution_); // todo
                    PathNodePtr pro_node =
                            dynamic ? expanded_nodes.find(pro_id, pro_t_id) : expanded_nodes.find(pro_id);
                    if (pro_node && pro_node->node_state == IN_CLOSE_SET && init_search) {
                        std::cout << "close" << std::endl;
                        continue;
                    }

                    // Check maximal velocity
                    Eigen::Vector3d pro_v = pro_state.tail(3);
                    if (fabs(pro_v(0)) > max_vel_ || fabs(pro_v(1)) > max_vel_ || fabs(pro_v(2)) > max_vel_) {
                        if (init_search) std::cout << "vel" << std::endl;
                        continue;
                    }

                    // Check not in the same voxel
                    Eigen::Vector3i diff = pro_id - cur_node->index;
                    int diff_time = pro_t_id - cur_node->time_idx;
                    if (diff.norm() == 0 && ((!dynamic) || diff_time == 0)) {
                        if (init_search) std::cout << "same" << std::endl;
                        continue;
                    }

                    // Check safety
                    Eigen::Vector3d pos;
                    Eigen::Matrix<double, 6, 1> xt;
                    bool is_occ = false;
                    for (int k = 1; k <= check_num_; ++k) {
                        double dt = tau * double(k) / double(check_num_);
                        stateTransit(cur_state, xt, input, dt);
                        pos = xt.head(3);
                        if (edt_environment_->sdf_map_->getInflateOccupancy(pos) == 1 ||
                            !edt_environment_->sdf_map_->isInBox(pos)) {
                            is_occ = true;
                            break;
                        }
                        if (!optimistic_ && edt_environment_->sdf_map_->getOccupancy(pos) == SDFMap::UNKNOWN) {
                            is_occ = true;
                            break;
                        }
                    }
                    if (is_occ) {
                        if (init_search) std::cout << "safe" << std::endl;
                        continue;
                    }

                    double tmp_time_to_goal, tmp_g_score, tmp_f_score;
                    tmp_g_score = (input.squaredNorm() + w_time_) * tau + cur_node->g_score;
                    tmp_f_score = tmp_g_score + lambda_heu_ * estimateHeuristic(pro_state, end_state, tmp_time_to_goal);

                    // Compare nodes expanded from the same parent
                    bool prune = false;
                    for (const PathNodePtr &expand_node: tmp_expand_nodes) {
                        if ((pro_id - expand_node->index).norm() == 0 &&
                            ((!dynamic) || pro_t_id == expand_node->time_idx)) {
                            prune = true;
                            if (tmp_f_score < expand_node->f_score) {
                                expand_node->f_score = tmp_f_score;
                                expand_node->g_score = tmp_g_score;
                                expand_node->state = pro_state;
                                expand_node->input = input;
                                expand_node->duration = tau;
                                if (dynamic) expand_node->time = cur_node->time + tau;
                            }
                            break;
                        }
                    }

                    if (prune) continue;

                    // This node end up in a voxel different from others
                    if (!pro_node) {
                        pro_node = std::make_shared<PathNode>();
                        pro_node->index = pro_id;
                        pro_node->state = pro_state;
                        pro_node->f_score = tmp_f_score;
                        pro_node->g_score = tmp_g_score;
                        pro_node->input = input;
                        pro_node->duration = tau;
                        pro_node->parent = cur_node;
                        pro_node->node_state = IN_OPEN_SET;
                        if (dynamic) {
                            pro_node->time = cur_node->time + tau;
                            pro_node->time_idx = timeToIndex(pro_node->time);
                        }
                        open_set.push(pro_node);

                        if (dynamic)
                            expanded_nodes.insert(pro_id, pro_node->time, pro_node);
                        else
                            expanded_nodes.insert(pro_id, pro_node);

                        tmp_expand_nodes.push_back(pro_node);

                        use_node_num += 1;
                        if (use_node_num == allocate_num_) {
                            cout << "run out of memory." << endl;
                            return NO_PATH;
                        }
                    } else if (pro_node->node_state == IN_OPEN_SET) {
                        if (tmp_g_score < pro_node->g_score) {
                            // pro_node->index = pro_id;
                            pro_node->state = pro_state;
                            pro_node->f_score = tmp_f_score;
                            pro_node->g_score = tmp_g_score;
                            pro_node->input = input;
                            pro_node->duration = tau;
                            pro_node->parent = cur_node;
                            if (dynamic) pro_node->time = cur_node->time + tau;
                        }
                    } else {
                        cout << "error type in searching: " << pro_node->node_state << endl;
                    }
                }
            }
        }

        cout << "open set empty, no path!" << endl;
        cout << "use node num: " << use_node_num << endl;
        cout << "iter num: " << iter_num << endl;
        return NO_PATH;
    }

    vector<PathNodePtr> KinodynamicAstar::retrievePath(PathNodePtr cur_node) {
        vector<PathNodePtr> path;
        while (cur_node) {
            path.push_back(cur_node);
            cur_node = cur_node->parent;
        }
        reverse(path.begin(), path.end());
        return path;
    }

    double KinodynamicAstar::estimateHeuristic(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2,
                                               double &optimal_time) const {
        const Vector3d dp = x2.head(3) - x1.head(3);
        const Vector3d v0 = x1.segment(3, 3);
        const Vector3d v1 = x2.segment(3, 3);

        double c1 = -36 * dp.dot(dp);
        double c2 = 24 * (v0 + v1).dot(dp);
        double c3 = -4 * (v0.dot(v0) + v0.dot(v1) + v1.dot(v1));
        double c4 = 0;
        double c5 = w_time_;

        std::vector<double> ts = quartic(c5, c4, c3, c2, c1);

        double v_max = max_vel_ * 0.5;
        double t_bar = (x1.head(3) - x2.head(3)).lpNorm<Infinity>() / v_max;
        ts.push_back(t_bar);

        double cost = 100000000;
        double t_d = t_bar;

        for (const double t: ts) {
            if (t < t_bar) continue;
            double c = -c1 / (3 * t * t * t) - c2 / (2 * t * t) - c3 / t + w_time_ * t;
            if (c < cost) {
                cost = c;
                t_d = t;
            }
        }

        optimal_time = t_d;

        return 1.0 * (1 + tie_breaker_) * cost;
    }

    bool KinodynamicAstar::computeShotTraj(const Eigen::VectorXd &state1, const Eigen::VectorXd &state2,
                                           const double time_to_goal,
                                           Eigen::MatrixXd &coef_shot) {
        /* ---------- get coefficient ---------- */
        const Vector3d p0 = state1.head(3);
        const Vector3d dp = state2.head(3) - p0;
        const Vector3d v0 = state1.segment(3, 3);
        const Vector3d v1 = state2.segment(3, 3);
        const Vector3d dv = v1 - v0;
        coef_shot = MatrixXd::Zero(3, 4);

        Vector3d a = 1.0 / 6.0 * (-12.0 / (time_to_goal * time_to_goal * time_to_goal) * (dp - v0 * time_to_goal) +
                                  6 / (time_to_goal * time_to_goal) * dv);
        Vector3d b = 0.5 * (6.0 / (time_to_goal * time_to_goal) * (dp - v0 * time_to_goal) - 2 / time_to_goal * dv);
        const Vector3d &c = v0;
        const Vector3d &d = p0;

        // 1/6 * alpha * t^3 + 1/2 * beta * t^2 + v0
        // a*t^3 + b*t^2 + v0*t + p0
        coef_shot.col(3) = a, coef_shot.col(2) = b, coef_shot.col(1) = c, coef_shot.col(0) = d;

        Vector3d coord, vel, acc;
        VectorXd poly1d, t, polyv, polya;
        Vector3i index;

        Eigen::MatrixXd Tm(4, 4);
        Tm << 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0;

        /* ---------- forward checking of trajectory ---------- */
        double t_delta = time_to_goal / 10;
        for (double time = t_delta; time <= time_to_goal; time += t_delta) {
            t = VectorXd::Zero(4);
            for (Eigen::Index j = 0; j < 4; j++)
                t(j) = pow(time, j);

            for (int dim = 0; dim < 3; dim++) {
                poly1d = coef_shot.row(dim);
                coord(dim) = poly1d.dot(t);
                vel(dim) = (Tm * poly1d).dot(t);
                acc(dim) = (Tm * Tm * poly1d).dot(t);

                if (fabs(vel(dim)) > max_vel_ || fabs(acc(dim)) > max_acc_) {
                    cout << "vel:" << vel(dim) << ", acc:" << acc(dim) << endl;
                    return false;
                }
            }

            if (coord(0) < origin_(0) || coord(0) >= map_size_3d_(0) ||
                coord(1) < origin_(1) || coord(1) >= map_size_3d_(1) ||
                coord(2) < origin_(2) || coord(2) >= map_size_3d_(2)) {
                return false;
            }

            if (edt_environment_->sdf_map_->getInflateOccupancy(coord) == SDFMap::OCCUPIED) {
                return false;
            }
        }
        return true;
    }

    vector<double> KinodynamicAstar::cubic(double a, double b, double c, double d) {
        vector<double> dts;

        double a2 = b / a;
        double a1 = c / a;
        double a0 = d / a;

        double Q = (3 * a1 - a2 * a2) / 9;
        double R = (9 * a1 * a2 - 27 * a0 - 2 * a2 * a2 * a2) / 54;
        double D = Q * Q * Q + R * R;
        if (D > 0) {
            double S = std::cbrt(R + sqrt(D));
            double T = std::cbrt(R - sqrt(D));
            dts.push_back(-a2 / 3 + (S + T));
            return dts;
        } else if (D == 0) {
            double S = std::cbrt(R);
            dts.push_back(-a2 / 3 + S + S);
            dts.push_back(-a2 / 3 - S);
            return dts;
        } else {
            double theta = acos(R / sqrt(-Q * Q * Q));
            dts.push_back(2 * sqrt(-Q) * cos(theta / 3) - a2 / 3);
            dts.push_back(2 * sqrt(-Q) * cos((theta + 2 * M_PI) / 3) - a2 / 3);
            dts.push_back(2 * sqrt(-Q) * cos((theta + 4 * M_PI) / 3) - a2 / 3);
            return dts;
        }
    }

    vector<double> KinodynamicAstar::quartic(double a, double b, double c, double d, double e) {
        vector<double> dts;

        double a3 = b / a;
        double a2 = c / a;
        double a1 = d / a;
        double a0 = e / a;

        vector<double> ys = cubic(1, -a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1 * a1 - a3 * a3 * a0);
        double y1 = ys.front();
        double r = a3 * a3 / 4 - a2 + y1;
        if (r < 0) return dts;

        double R = sqrt(r);
        double D, E;
        if (R != 0) {
            D = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
            E = sqrt(0.75 * a3 * a3 - R * R - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3 * a3 * a3) / R);
        } else {
            D = sqrt(0.75 * a3 * a3 - 2 * a2 + 2 * sqrt(y1 * y1 - 4 * a0));
            E = sqrt(0.75 * a3 * a3 - 2 * a2 - 2 * sqrt(y1 * y1 - 4 * a0));
        }

        if (!std::isnan(D)) {
            dts.push_back(-a3 / 4 + R / 2 + D / 2);
            dts.push_back(-a3 / 4 + R / 2 - D / 2);
        }
        if (!std::isnan(E)) {
            dts.push_back(-a3 / 4 - R / 2 + E / 2);
            dts.push_back(-a3 / 4 - R / 2 - E / 2);
        }

        return dts;
    }

    std::vector<Eigen::Vector3d> KinodynamicAstar::getKinoTraj(const vector<PathNodePtr> &path,
                                                               const bool is_shot_succ, const Eigen::MatrixXd &coef_shot, const double t_shot) {
        const double delta_t = 0.01;

        vector<Vector3d> state_list;

        /* ---------- get traj of searching ---------- */
        for (const PathNodePtr &node: path) {
            Vector3d input = node->input;
            double duration = node->duration;
            Matrix<double, 6, 1> x0 = node->state, xt;
            for (double t = 0; t < duration; t += delta_t) {
                stateTransit(x0, xt, input, t);
                state_list.emplace_back(xt.head(3));
            }
        }

        /* ---------- get traj of one shot ---------- */
        if (is_shot_succ) {
            Vector3d coord;
            VectorXd poly1d, time(4);

            for (double t = delta_t; t <= t_shot; t += delta_t) {
                for (Eigen::Index j = 0; j < 4; j++)
                    time(j) = pow(t, j);

                for (int dim = 0; dim < 3; dim++) {
                    poly1d = coef_shot.row(dim);
                    coord(dim) = poly1d.dot(time);
                }
                state_list.push_back(coord);
            }
        }

        return state_list;
    }

    void KinodynamicAstar::getSamples(const vector<PathNodePtr> &path,
                                      const Eigen::Vector3d &start_v, const Eigen::Vector3d &end_v,
                                      const bool is_shot_succ, const Eigen::MatrixXd &coef_shot, const double t_shot,
                                      double &ts, vector<Eigen::Vector3d> &point_set,
                                      vector<Eigen::Vector3d> &start_end_derivatives) {
        /* ---------- path duration ---------- */

        double sum_t = 0;
        for (const PathNodePtr &node: path) {
            sum_t += node->duration;
        }
        if (is_shot_succ) {
            sum_t += t_shot;
        }

        int seg_num = floor(sum_t / ts);
        seg_num = max(8, seg_num);
        ts = sum_t / seg_num;
        double t_from_pre_node = ts;

        for (const PathNodePtr &node: path) {
            Eigen::Matrix<double, 6, 1> x0 = node->state, xt;
            Vector3d input = node->input;
            while (t_from_pre_node < node->duration) {
                stateTransit(x0, xt, input, t_from_pre_node);
                point_set.emplace_back(xt.head(3));
                t_from_pre_node += ts;
            }
            t_from_pre_node -= node->duration;
        }

        if (is_shot_succ) {
            while (t_from_pre_node < t_shot) {
                Vector3d coord;
                Vector4d poly1d, time;

                for (Eigen::Index j = 0; j < 4; j++)
                    time(j) = pow(t_from_pre_node, j);

                for (int dim = 0; dim < 3; dim++) {
                    poly1d = coef_shot.row(dim);
                    coord(dim) = poly1d.dot(time);
                }

                point_set.push_back(coord);
                t_from_pre_node += ts;
            }
        }

        // Calculate boundary vel and acc
        Eigen::Vector3d end_vel, end_acc;
        if (is_shot_succ) {
            end_vel = end_v;
            for (int dim = 0; dim < 3; ++dim) {
                Vector4d coe = coef_shot.row(dim);
                end_acc(dim) = 2 * coe(2) + 6 * coe(3) * t_shot;
            }
        } else {
            end_vel = path.back()->state.tail(3);
            end_acc = path.back()->input;
        }

        // calculate start acc
        Eigen::Vector3d start_acc;
        if (path.empty()) {
            // no searched traj, calculate by shot traj
            start_acc = 2 * coef_shot.col(2);
            cout << "path nodes empty\n";
        } else {
            // input of searched traj
            start_acc = path.front()->input;
            cout << "path nodes not empty\n";
        }

        start_end_derivatives.push_back(start_v);
        start_end_derivatives.push_back(end_vel);
        start_end_derivatives.push_back(start_acc);
        start_end_derivatives.push_back(end_acc);

        cout << "get sample start end derivatives\n";
        for (auto d: start_end_derivatives) {
            cout << d.transpose() << endl;
        }
    }

    Eigen::Vector3i KinodynamicAstar::posToIndex(const Eigen::Vector3d &pt) {
        Vector3i idx = ((pt - origin_) / resolution_).array().floor().cast<int>();
        return idx;
    }

    int KinodynamicAstar::timeToIndex(double time) const {
        return floor((time - time_origin_) / time_resolution_);
    }

    void KinodynamicAstar::stateTransit(Eigen::Matrix<double, 6, 1> &state0,
                                        Eigen::Matrix<double, 6, 1> &state1, const Eigen::Vector3d &um,
                                        double tau) {
        Eigen::Matrix<double, 6, 6> phi = Eigen::MatrixXd::Identity(6, 6);  // state transit matrix
        for (Eigen::Index i = 0; i < 3; ++i)
            phi(i, i + 3) = tau;

        Eigen::Matrix<double, 6, 1> integral;
        integral.head(3) = 0.5 * pow(tau, 2) * um;
        integral.tail(3) = tau * um;

        state1 = phi * state0 + integral;
    }

}  // namespace fast_planner
