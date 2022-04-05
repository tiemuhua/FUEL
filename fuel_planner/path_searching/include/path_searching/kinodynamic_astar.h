#ifndef _KINODYNAMIC_ASTAR_H
#define _KINODYNAMIC_ASTAR_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <ros/console.h>
#include <ros/ros.h>
#include <utility>
#include <string>
#include <unordered_map>
#include "plan_env/edt_environment.h"
#include <boost/functional/hash.hpp>
#include <queue>
#include <path_searching/matrix_hash.h>

namespace fast_planner {
// #define REACH_HORIZON 1
// #define REACH_END 2
// #define NO_PATH 3
#define IN_CLOSE_SET 'a'
#define IN_OPEN_SET 'b'
#define NOT_EXPAND 'c'
#define inf 1 >> 30

    class PathNode {
    public:
        /* -------------------- */
        Eigen::Vector3i index;
        Eigen::Matrix<double, 6, 1> state;
        double g_score{}, f_score{};
        Eigen::Vector3d input;
        double duration{};
        double time{};  // dyn
        int time_idx{};
        shared_ptr<PathNode> parent;
        char node_state;

        /* -------------------- */
        PathNode() {
            parent = nullptr;
            node_state = NOT_EXPAND;
            input = Eigen::Vector3d::Zero();
        }

        ~PathNode() = default;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    typedef shared_ptr<PathNode> PathNodePtr;

    class NodeComparator {
    public:
        bool operator()(const PathNodePtr& node1, const PathNodePtr& node2) {
            return node1->f_score > node2->f_score;
        }
    };

    class NodeHashTable {
    private:
        /* data */
        std::unordered_map<Eigen::Vector3i, PathNodePtr, matrix_hash<Eigen::Vector3i>> data_3d_;
        std::unordered_map<Eigen::Vector4i, PathNodePtr, matrix_hash<Eigen::Vector4i>> data_4d_;

    public:
        NodeHashTable(/* args */) = default;

        ~NodeHashTable() = default;

        void insert(const Eigen::Vector3i& idx, const PathNodePtr& node) {
            data_3d_.insert(std::make_pair(idx, node));
        }

        void insert(Eigen::Vector3i idx, int time_idx, const PathNodePtr& node) {
            data_4d_.insert(std::make_pair(Eigen::Vector4i(idx(0), idx(1), idx(2), time_idx), node));
        }

        PathNodePtr find(const Eigen::Vector3i& idx) {
            auto iter = data_3d_.find(idx);
            return iter == data_3d_.end() ? nullptr : iter->second;
        }

        PathNodePtr find(Eigen::Vector3i idx, int time_idx) {
            auto iter = data_4d_.find(Eigen::Vector4i(idx(0), idx(1), idx(2), time_idx));
            return iter == data_4d_.end() ? nullptr : iter->second;
        }

        void clear() {
            data_3d_.clear();
            data_4d_.clear();
        }
    };

    class KinodynamicAstar {
    private:
        /* ---------- main data structure ---------- */
        vector<PathNodePtr> path_node_pool_;
        int use_node_num_, iter_num_;
        std::vector<PathNodePtr> path_nodes_;

        /* ---------- record data ---------- */
        Eigen::Vector3d start_vel_, end_vel_;
        EDTEnvironment::Ptr edt_environment_;
        bool is_shot_succ_ = false;
        Eigen::MatrixXd coef_shot_;
        double t_shot_{};

        /* ---------- parameter ---------- */
        /* search */
        double max_tau_{}, init_max_tau_{};
        double max_vel_{}, max_acc_{};
        double w_time_{}, horizon_{}, lambda_heu_{};
        int allocate_num_{}, check_num_{};
        double tie_breaker_;
        bool optimistic_{};

        /* map */
        double resolution_{}, time_resolution_{};
        Eigen::Vector3d origin_, map_size_3d_;
        double time_origin_{};

        /* helper */
        Eigen::Vector3i posToIndex(const Eigen::Vector3d& pt);

        int timeToIndex(double time) const;

        static vector<PathNodePtr> retrievePath(PathNodePtr cur_node);

        /* shot trajectory */
        static vector<double> cubic(double a, double b, double c, double d);

        static vector<double> quartic(double a, double b, double c, double d, double e);

        bool computeShotTraj(Eigen::VectorXd state1, Eigen::VectorXd state2, double time_to_goal);

        double estimateHeuristic(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, double &optimal_time) const;

        /* state propagation */
        static void stateTransit(Eigen::Matrix<double, 6, 1> &state0, Eigen::Matrix<double, 6, 1> &state1,
                          const Eigen::Vector3d& um, double tau);

    public:
        KinodynamicAstar(ros::NodeHandle &nh, const EDTEnvironment::Ptr &env);

        enum {
            REACH_HORIZON = 1, REACH_END = 2, NO_PATH = 3, NEAR_END = 4
        };

        /* main API */
        void reset();

        int search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& start_v, const Eigen::Vector3d& start_a,
                   const Eigen::Vector3d& end_pt, const Eigen::Vector3d& end_v, bool init, bool dynamic = false,
                   double time_start = -1.0);

        std::vector<Eigen::Vector3d> getKinoTraj();

        void getSamples(double &ts, vector<Eigen::Vector3d> &point_set,
                        vector<Eigen::Vector3d> &start_end_derivatives);

        typedef shared_ptr<KinodynamicAstar> Ptr;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}  // namespace fast_planner

#endif