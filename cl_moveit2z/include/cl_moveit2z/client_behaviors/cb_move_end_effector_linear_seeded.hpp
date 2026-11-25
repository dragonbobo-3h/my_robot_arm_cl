// Copyright 2021 RobosoftAI Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <vector>
#include <optional>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <smacc2/smacc_asynchronous_client_behavior.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <fstream>

#include <cl_moveit2z/cl_moveit2z.hpp>

namespace cl_moveit2z
{

/**
 * @brief CbMoveEndEffectorLinearSeeded
 *
 * Moves from current EE pose to target pose using linear Cartesian interpolation
 * in task space and computes a single cartesian trajectory for one-shot execution.
 * computeCartesianPath internally propagates along waypoints using sequential IK,
 * equivalent to using prior points as seeds.
 */
class CbMoveEndEffectorLinearSeeded : public smacc2::SmaccAsyncClientBehavior
{
public:
  geometry_msgs::msg::PoseStamped targetPose;
  std::string tip_link_;
  double planningTimeSec_{1.0};
  double linearStepMeters_{0.01};
  int maxSegments_{200};
  double jumpThreshold_{0.0};
  double maxJointStepRad_{1.5};
  double maxJointTotalDeltaRad_{6.2832};
  double minPathFraction_{0.98};
  std::optional<double> velocityScaling_;  // Optional velocity scaling (resets inherited value)

  CbMoveEndEffectorLinearSeeded() = default;

  CbMoveEndEffectorLinearSeeded(
    geometry_msgs::msg::PoseStamped target_pose,
    std::string tip_link = "",
    double linear_step_meters = 0.01,
    double planning_time_sec = 1.0)
  : targetPose(target_pose),
    tip_link_(tip_link),
    planningTimeSec_(planning_time_sec),
    linearStepMeters_(linear_step_meters)
  {
  }

  void onEntry() override
  {
    this->requiresClient(movegroupClient_);
    auto & mgi = *(movegroupClient_->moveGroupClientInterface);

    // Auto-load cartesian motion parameters from YAML (if available)
    // This allows tuning velocity/trajectory params without recompilation
    loadConfigFromYaml("cl_moveit2z", "config/cartesian.yaml");
    loadSpeedOverrideFromYaml("je_arm_pcb_inspection_sm", "config/motion_speed.yaml");

    // Reset MoveGroup velocity scaling to prevent inheritance from prior joint moves
    if (velocityScaling_)
    {
      mgi.setMaxVelocityScalingFactor(*velocityScaling_);
    }
    else
    {
      mgi.setMaxVelocityScalingFactor(1.0);  // Reset to 100% by default
    }

    if (isShutdownRequested())
    {
      RCLCPP_WARN(getLogger(), "[CbMoveEndEffectorLinearSeeded] canceled before planning");
      return;
    }

    if (linearStepMeters_ <= 0.0)
    {
      linearStepMeters_ = 0.01;
    }

    mgi.setPlanningTime(planningTimeSec_);
    mgi.setPoseReferenceFrame(targetPose.header.frame_id);

    const std::string tipLink = tip_link_.empty() ? mgi.getEndEffectorLink() : tip_link_;
    const auto currentPose = mgi.getCurrentPose(tipLink);

    const double dx = targetPose.pose.position.x - currentPose.pose.position.x;
    const double dy = targetPose.pose.position.y - currentPose.pose.position.y;
    const double dz = targetPose.pose.position.z - currentPose.pose.position.z;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Let MoveIt's computeCartesianPath perform a single uniform interpolation
    // directly to targetPose.pose. Pre-subdividing at linearStepMeters_ caused
    // MoveIt to re-subdivide every segment in half when pick_ik stops within
    // its 1 mm tolerance (doubling waypoint count and slowing IPTP/TOTG by >2x).
    double effectiveStep = linearStepMeters_;
    if (maxSegments_ > 0 && distance > 0.0)
    {
      effectiveStep = std::max(effectiveStep, distance / static_cast<double>(maxSegments_));
    }

    RCLCPP_INFO(
      getLogger(),
      "[CbMoveEndEffectorLinearSeeded] linear move start -> target, distance=%.4f m, effectiveStep=%.4f m, velScale=%.3f",
      distance, effectiveStep, velocityScaling_.value_or(1.0));

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(targetPose.pose);

    if (isShutdownRequested())
    {
      RCLCPP_WARN(getLogger(), "[CbMoveEndEffectorLinearSeeded] canceled before cartesian path computation");
      return;
    }

    mgi.setStartStateToCurrentState();
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double pathFraction =
      mgi.computeCartesianPath(waypoints, effectiveStep, jumpThreshold_, trajectory, true);

    if (pathFraction < minPathFraction_)
    {
      if (!isShutdownRequested())
      {
        RCLCPP_WARN(
          getLogger(),
          "[CbMoveEndEffectorLinearSeeded] cartesian path incomplete: fraction=%.3f (< %.3f)",
          pathFraction, minPathFraction_);
        movegroupClient_->postEventMotionExecutionFailed();
        this->postFailureEvent();
      }
      return;
    }

    if (!validateTrajectoryJointExcursion(trajectory))
    {
      if (!isShutdownRequested())
      {
        movegroupClient_->postEventMotionExecutionFailed();
        this->postFailureEvent();
      }
      return;
    }

    applyCartesianVelocityScaling(trajectory);

    if (isShutdownRequested())
    {
      RCLCPP_WARN(getLogger(), "[CbMoveEndEffectorLinearSeeded] canceled before trajectory execution");
      return;
    }

    const bool execOk = executeTrajectoryInterruptible(mgi, trajectory);
    if (!execOk)
    {
      if (!isShutdownRequested())
      {
        RCLCPP_WARN(getLogger(), "[CbMoveEndEffectorLinearSeeded] trajectory execution failed");
        movegroupClient_->postEventMotionExecutionFailed();
        this->postFailureEvent();
      }
      return;
    }

    if (isShutdownRequested())
    {
      RCLCPP_WARN(
        getLogger(),
        "[CbMoveEndEffectorLinearSeeded] canceled after execute (no success/failure event will be posted)");
      return;
    }

    movegroupClient_->postEventMotionExecutionSucceded();
    this->postSuccessEvent();
  }

  /**
   * Load cartesian motion parameters from YAML file.
   * YAML format:
   *   linear_step_meters: 0.01
   *   planning_time_sec: 1.0
   *   max_segments: 200
   *   jump_threshold: 0.0
   *   min_path_fraction: 0.98
   *   velocity_scaling: 1.0  # optional - sets scaling (1.0 = 100%)
   */
  void loadConfigFromYaml(const std::string & pkg, const std::string & filepath)
  {
    std::string pkgpath = ament_index_cpp::get_package_share_directory(pkg);

    if (pkgpath.empty())
    {
      RCLCPP_ERROR_STREAM(
        getLogger(), "[CbMoveEndEffectorLinearSeeded] package not found: " << pkg);
      return;
    }

    std::string fullpath = pkgpath + "/" + filepath;

    if (!std::filesystem::exists(fullpath))
    {
      RCLCPP_WARN_STREAM(getLogger(), "[CbMoveEndEffectorLinearSeeded] config file not found: " << fullpath);
      return;
    }

    try
    {
      std::ifstream ifs(fullpath);
      YAML::Node node = YAML::Load(ifs);

      if (node["linear_step_meters"])
      {
        linearStepMeters_ = node["linear_step_meters"].as<double>();
      }
      if (node["planning_time_sec"])
      {
        planningTimeSec_ = node["planning_time_sec"].as<double>();
      }
      if (node["max_segments"])
      {
        maxSegments_ = node["max_segments"].as<int>();
      }
      if (node["jump_threshold"])
      {
        jumpThreshold_ = node["jump_threshold"].as<double>();
      }
      if (node["max_joint_step_rad"])
      {
        maxJointStepRad_ = node["max_joint_step_rad"].as<double>();
      }
      if (node["max_joint_total_delta_rad"])
      {
        maxJointTotalDeltaRad_ = node["max_joint_total_delta_rad"].as<double>();
      }
      if (node["min_path_fraction"])
      {
        minPathFraction_ = node["min_path_fraction"].as<double>();
      }
      if (node["velocity_scaling"])
      {
        velocityScaling_ = node["velocity_scaling"].as<double>();
      }

      RCLCPP_INFO(
        getLogger(),
        "[CbMoveEndEffectorLinearSeeded] loaded config from %s: step=%.4f, planTime=%.2f, maxSegs=%d, jumpThresh=%.4f, maxJointStep=%.3f, maxJointTotal=%.3f, minPathFrac=%.3f, velScale=%s",
        fullpath.c_str(),
        linearStepMeters_,
        planningTimeSec_,
        maxSegments_,
        jumpThreshold_,
        maxJointStepRad_,
        maxJointTotalDeltaRad_,
        minPathFraction_,
        velocityScaling_ ? std::to_string(*velocityScaling_).c_str() : "default(1.0)");
    }
    catch (const YAML::Exception & ex)
    {
      RCLCPP_ERROR_STREAM(
        getLogger(),
        "[CbMoveEndEffectorLinearSeeded] YAML error loading " << fullpath << ": " << ex.what());
    }
  }

  void loadSpeedOverrideFromYaml(const std::string & pkg, const std::string & filepath)
  {
    std::string pkgpath;
    try
    {
      pkgpath = ament_index_cpp::get_package_share_directory(pkg);
    }
    catch (const std::exception &)
    {
      return;
    }

    const std::string fullpath = pkgpath + "/" + filepath;
    if (!std::filesystem::exists(fullpath))
    {
      return;
    }

    try
    {
      const YAML::Node node = YAML::LoadFile(fullpath);
      const double globalScale =
        node["global_speed_scale"] ? node["global_speed_scale"].as<double>() : 1.0;
      const double cartScale =
        node["cartesian_velocity_scaling"]
          ? node["cartesian_velocity_scaling"].as<double>()
          : velocityScaling_.value_or(1.0);
      velocityScaling_ = std::clamp(cartScale * globalScale, 0.05, 1.0);
    }
    catch (const YAML::Exception & ex)
    {
      RCLCPP_WARN_STREAM(
        getLogger(),
        "[CbMoveEndEffectorLinearSeeded] YAML error loading " << fullpath << ": " << ex.what());
    }
  }

protected:
  void applyCartesianVelocityScaling(moveit_msgs::msg::RobotTrajectory & trajectory) const
  {
    // Strip TOTG corner accelerations so JTC uses smooth cubic Hermite splines
    // instead of quintic splines that ring between discontinuous +/-a_max corners.
    for (auto & pt : trajectory.joint_trajectory.points)
    {
      pt.accelerations.clear();
    }

    const double scale = velocityScaling_.value_or(1.0);
    if (scale >= 0.9999 || scale <= 0.0)
    {
      return;
    }

    // MoveIt2 Humble GetCartesianPath has no velocity_scaling field; rescale
    // time_from_start and velocities directly (q'(t') = q(s*t')).
    const double invScale = 1.0 / scale;
    for (auto & pt : trajectory.joint_trajectory.points)
    {
      const double tSec =
        static_cast<double>(pt.time_from_start.sec) +
        static_cast<double>(pt.time_from_start.nanosec) * 1e-9;
      const double scaledSec = tSec * invScale;
      const auto totalNs = static_cast<int64_t>(std::llround(scaledSec * 1e9));
      pt.time_from_start.sec = static_cast<int32_t>(totalNs / 1000000000LL);
      pt.time_from_start.nanosec = static_cast<uint32_t>(totalNs % 1000000000LL);
      for (double & v : pt.velocities)
      {
        v *= scale;
      }
    }
  }

  bool validateTrajectoryJointExcursion(
    const moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    const auto & points = trajectory.joint_trajectory.points;
    if (points.size() < 2)
    {
      return true;
    }

    const size_t nJoints = points.front().positions.size();
    std::vector<double> cumulativeDelta(nJoints, 0.0);

    for (size_t i = 1; i < points.size(); ++i)
    {
      const auto & prev = points[i - 1].positions;
      const auto & curr = points[i].positions;
      if (prev.size() != nJoints || curr.size() != nJoints)
      {
        continue;
      }

      for (size_t j = 0; j < nJoints; ++j)
      {
        const double stepDelta = std::abs(curr[j] - prev[j]);
        cumulativeDelta[j] += stepDelta;

        const std::string jointName =
          (j < trajectory.joint_trajectory.joint_names.size())
            ? trajectory.joint_trajectory.joint_names[j]
            : ("joint_" + std::to_string(j));

        if (maxJointStepRad_ > 0.0 && stepDelta > maxJointStepRad_)
        {
          RCLCPP_ERROR(
            getLogger(),
            "[CbMoveEndEffectorLinearSeeded] REJECTED trajectory: %s jumped %.3f rad between waypoint %zu and %zu (limit=%.3f rad)",
            jointName.c_str(), stepDelta, i - 1, i, maxJointStepRad_);
          return false;
        }

        if (maxJointTotalDeltaRad_ > 0.0 && cumulativeDelta[j] > maxJointTotalDeltaRad_)
        {
          RCLCPP_ERROR(
            getLogger(),
            "[CbMoveEndEffectorLinearSeeded] REJECTED trajectory: %s cumulative travel %.3f rad exceeded limit %.3f rad at waypoint %zu",
            jointName.c_str(), cumulativeDelta[j], maxJointTotalDeltaRad_, i);
          return false;
        }
      }
    }

    return true;
  }

  bool executeTrajectoryInterruptible(
    moveit::planning_interface::MoveGroupInterface & mgi,
    const moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    auto execFuture =
      std::async(std::launch::async, [&mgi, &trajectory]() { return mgi.execute(trajectory); });

    bool stopRequested = false;
    while (execFuture.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready)
    {
      if (isShutdownRequested() && !stopRequested)
      {
        RCLCPP_WARN(
          getLogger(),
          "[CbMoveEndEffectorLinearSeeded] PAUSE/CANCEL requested during execute -> stop() sent");
        mgi.stop();
        stopRequested = true;
      }
    }

    const auto execResult = execFuture.get();
    return (execResult == moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  }

  ClMoveit2z * movegroupClient_ = nullptr;
};

}  // namespace cl_moveit2z
