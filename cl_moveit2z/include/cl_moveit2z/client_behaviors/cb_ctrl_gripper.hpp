#pragma once

#include <cctype>
#include <chrono>
#include <memory>
#include <string>

#include <cmath>

#include <algorithm>

#include <common/msg/oculus_init_joint_state.hpp>
#include <je_software/msg/end_effector_command.hpp>
#include <je_software/msg/end_effector_command_lr.hpp>
#include <rclcpp/rclcpp.hpp>
#include <smacc2/smacc_asynchronous_client_behavior.hpp>

namespace cl_moveit2z
{
namespace detail
{
inline bool getBoolParameter(
  const rclcpp::Node::SharedPtr & node, const std::string & name, bool defaultValue)
{
  rclcpp::Parameter parameter;
  if (!node->get_parameter(name, parameter))
  {
    return defaultValue;
  }

  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL)
  {
    return parameter.as_bool();
  }

  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
  {
    auto value = parameter.as_string();
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "true" || value == "1" || value == "yes" || value == "on")
    {
      return true;
    }
    if (value == "false" || value == "0" || value == "no" || value == "off")
    {
      return false;
    }
  }

  return defaultValue;
}

inline double getDoubleParameter(
  const rclcpp::Node::SharedPtr & node, const std::string & name, double defaultValue)
{
  rclcpp::Parameter parameter;
  if (!node->get_parameter(name, parameter))
  {
    return defaultValue;
  }

  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
  {
    return parameter.as_double();
  }

  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
  {
    return static_cast<double>(parameter.as_int());
  }

  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
  {
    try
    {
      return std::stod(parameter.as_string());
    }
    catch (const std::exception &)
    {
      return defaultValue;
    }
  }

  return defaultValue;
}

inline std::string getStringParameter(
  const rclcpp::Node::SharedPtr & node, const std::string & name,
  const std::string & defaultValue)
{
  rclcpp::Parameter parameter;
  if (node->get_parameter(name, parameter) &&
    parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
  {
    return parameter.as_string();
  }
  return defaultValue;
}
}  // namespace detail

class CbCtrlGripper : public smacc2::SmaccAsyncClientBehavior
{
public:
  CbCtrlGripper() = default;

  CbCtrlGripper(
    int mode, double position, int preset = 0, bool leftValid = true, bool rightValid = false,
    std::string topic = "/end_effector_cmd_lr", double timeoutSec = 2.0,
    std::string feedbackTopic = "/joint_states_double_arm", double positionTolerance = 0.03,
    std::string command = "", double torque = 0.0, bool waitForFeedback = true,
    double commandDelaySec = -1.0)
  : mode_(mode),
    position_(position),
    preset_(preset),
    left_valid_(leftValid),
    right_valid_(rightValid),
    topic_(std::move(topic)),
    timeout_sec_(timeoutSec),
    feedback_topic_(std::move(feedbackTopic)),
    position_tolerance_(positionTolerance),
    command_(normalizeCommand(command)),
    torque_(torque),
    wait_for_feedback_(waitForFeedback),
    command_delay_sec_(commandDelaySec)
  {
  }

  static CbCtrlGripper Position(
    double position, bool leftValid = true, bool rightValid = false,
    std::string topic = "/end_effector_cmd_lr",
    double timeoutSec = 2.0, std::string feedbackTopic = "/joint_states_double_arm",
    double positionTolerance = 0.03)
  {
    return CbCtrlGripper(
      je_software::msg::EndEffectorCommand::MODE_POSITION, position, 0, leftValid, rightValid,
      std::move(topic), timeoutSec, std::move(feedbackTopic), positionTolerance);
  }

  static CbCtrlGripper Preset(
    int preset, bool leftValid = true, bool rightValid = false,
    std::string topic = "/end_effector_cmd_lr",
    double timeoutSec = 2.0, std::string feedbackTopic = "/joint_states_double_arm",
    double positionTolerance = 0.03)
  {
    return CbCtrlGripper(
      je_software::msg::EndEffectorCommand::MODE_PRESET, 0.0, preset, leftValid, rightValid,
      std::move(topic), timeoutSec, std::move(feedbackTopic), positionTolerance);
  }

  static CbCtrlGripper Torque(
    std::string command, double torque, bool leftValid = true, bool rightValid = false,
    std::string topic = "/end_effector_cmd_lr", double timeoutSec = 2.0,
    std::string feedbackTopic = "/joint_states_double_arm", double positionTolerance = 0.03,
    bool waitForFeedback = true)
  {
    return CbCtrlGripper(
      je_software::msg::EndEffectorCommand::MODE_TORQUE, 0.0, 0, leftValid, rightValid,
      std::move(topic), timeoutSec, std::move(feedbackTopic), positionTolerance,
      std::move(command), torque, waitForFeedback);
  }

  void onEntry() override
  {
    command_completed_ = false;
    waiting_feedback_ = false;
    success_pending_ = false;
    enable_gripper_control_ = true;
    simulated_action_duration_sec_ = 2.0;
    gripper_action_delay_sec_ = 0.0;

    auto node = getNode();
    enable_gripper_control_ = detail::getBoolParameter(node, "enable_gripper_control", true);
    simulated_action_duration_sec_ = detail::getDoubleParameter(node, "simulated_gripper_action_sec", 2.0);
    gripper_action_delay_sec_ = detail::getDoubleParameter(node, "gripper_action_delay_sec", 0.0);

    const auto controlledArm = detail::getStringParameter(node, "controlled_arm", "both");
    if (controlledArm == "left")
    {
      right_valid_ = false;
    }
    else if (controlledArm == "right")
    {
      left_valid_ = false;
    }

    if (!left_valid_ && !right_valid_)
    {
      RCLCPP_INFO(
        getLogger(),
        "[CbCtrlGripper] Command skipped because it only targets the blocked arm (controlled_arm=%s).",
        controlledArm.c_str());
      scheduleSuccess(position_, position_, 0.0);
      return;
    }

    if (
      mode_ != je_software::msg::EndEffectorCommand::MODE_POSITION &&
      mode_ != je_software::msg::EndEffectorCommand::MODE_PRESET &&
      mode_ != je_software::msg::EndEffectorCommand::MODE_TORQUE)
    {
      RCLCPP_WARN(getLogger(), "[CbCtrlGripper] Invalid gripper mode: %d", mode_);
      markFailure("Invalid gripper mode");
      return;
    }

    if (mode_ == je_software::msg::EndEffectorCommand::MODE_TORQUE)
    {
      if (!isValidTorqueCommand(command_))
      {
        RCLCPP_WARN(getLogger(), "[CbCtrlGripper] Invalid torque command: '%s'", command_.c_str());
        markFailure("Invalid torque command");
        return;
      }

      if (!std::isfinite(torque_) || torque_ <= 0.0)
      {
        RCLCPP_WARN(getLogger(), "[CbCtrlGripper] Invalid torque value: %.4f", torque_);
        markFailure("Invalid torque value");
        return;
      }
    }

    if (timeout_sec_ <= 0.0)
    {
      timeout_sec_ = 2.0;
    }

    if (position_tolerance_ <= 0.0)
    {
      position_tolerance_ = 0.03;
    }

    if (!enable_gripper_control_)
    {
      const double effectiveDelay = getEffectiveCompletionDelaySec();
      RCLCPP_INFO(
        getLogger(),
        "[CbCtrlGripper] enable_gripper_control=false, simulating gripper action for %.2fs",
        effectiveDelay);
      scheduleSuccess(position_, position_, effectiveDelay);
      return;
    }

    if (!publisher_)
    {
      publisher_ =
        node->create_publisher<je_software::msg::EndEffectorCommandLR>(topic_, rclcpp::QoS(10).reliable());
    }

    if (publisher_->get_subscription_count() == 0)
    {
      if (!wait_for_feedback_)
      {
        RCLCPP_ERROR(
          getLogger(),
          "[CbCtrlGripper] No subscribers on %s in open-loop mode; refusing to report success.",
          topic_.c_str());
        markFailure("No gripper command subscribers in open-loop mode");
        return;
      }

      RCLCPP_WARN(
        getLogger(),
        "[CbCtrlGripper] Topic %s has no subscribers at command dispatch time. The gripper command may be ignored.",
        topic_.c_str());
    }

    if (wait_for_feedback_ && !feedback_sub_)
    {
      feedback_sub_ = node->create_subscription<common::msg::OculusInitJointState>(
        feedback_topic_,
        rclcpp::QoS(10).reliable(),
        [this](const common::msg::OculusInitJointState::SharedPtr msg)
        {
          onFeedback(msg);
        });
    }

    if (wait_for_feedback_ && feedback_topic_ != legacy_feedback_topic_ && !feedback_sub_legacy_)
    {
      feedback_sub_legacy_ = node->create_subscription<common::msg::OculusInitJointState>(
        legacy_feedback_topic_,
        rclcpp::QoS(10).reliable(),
        [this](const common::msg::OculusInitJointState::SharedPtr msg)
        {
          onFeedback(msg);
        });
    }

    je_software::msg::EndEffectorCommandLR msg;
    msg.left_valid = left_valid_;
    msg.right_valid = right_valid_;

    auto fill_command = [&](je_software::msg::EndEffectorCommand & command)
    {
      command.mode = mode_;
      command.position = position_;
      command.preset = preset_;
      command.command = command_;
      command.torque = torque_;
    };

    if (left_valid_)
    {
      fill_command(msg.left);
    }
    if (right_valid_)
    {
      fill_command(msg.right);
    }

    publisher_->publish(msg);

    command_sent_time_ = node->now();
    waiting_feedback_ = true;

    if (!watchdog_timer_)
    {
      watchdog_timer_ = node->create_wall_timer(
        std::chrono::milliseconds(50),
        [this]()
        {
          onWatchdogTick();
        });
    }

    RCLCPP_INFO(
      getLogger(),
      "[CbCtrlGripper] Published cmd topic=%s mode=%d left_valid=%d right_valid=%d position=%.4f "
      "preset=%d command=%s torque=%.4f wait_for_feedback=%d command_delay_sec=%.2f feedback_topic=%s fallback_feedback_topic=%s timeout=%.2fs tol=%.4f",
      topic_.c_str(), mode_, left_valid_, right_valid_, position_, preset_, command_.c_str(), torque_,
      wait_for_feedback_, command_delay_sec_, feedback_topic_.c_str(), legacy_feedback_topic_.c_str(), timeout_sec_,
      position_tolerance_);

    if (!wait_for_feedback_)
    {
      RCLCPP_INFO(
        getLogger(),
        "[CbCtrlGripper] Open-loop mode enabled; command will be considered successful after %.2fs without feedback.",
        timeout_sec_);
    }

  }

  void onExit() override
  {
    waiting_feedback_ = false;
    if (completion_timer_)
    {
      completion_timer_->cancel();
      completion_timer_.reset();
    }
    if (watchdog_timer_)
    {
      watchdog_timer_->cancel();
      watchdog_timer_.reset();
    }
  }

private:
  void onWatchdogTick()
  {
    if (!waiting_feedback_ || command_completed_)
    {
      return;
    }

    const auto elapsed = (getNode()->now() - command_sent_time_).seconds();
    if (elapsed > timeout_sec_)
    {
      if (!wait_for_feedback_)
      {
        scheduleSuccess(position_, position_);
      }
      else if (mode_ == je_software::msg::EndEffectorCommand::MODE_POSITION)
      {
        markFailure("Timed out waiting for gripper position feedback");
      }
      else
      {
        // Preset mode cannot be inferred precisely from position feedback; keep timeout success fallback.
        scheduleSuccess(position_, position_);
      }
    }
  }

  void onFeedback(const common::msg::OculusInitJointState::SharedPtr msg)
  {
    if (!msg || !waiting_feedback_ || command_completed_)
    {
      return;
    }

    const bool leftOk = !left_valid_ || (msg->left_valid && isTargetReached(msg->left_gripper));
    const bool rightOk = !right_valid_ || (msg->right_valid && isTargetReached(msg->right_gripper));

    if (leftOk && rightOk)
    {
      scheduleSuccess(msg->left_gripper, msg->right_gripper);
    }
  }

  double getEffectiveCompletionDelaySec() const
  {
    if (command_delay_sec_ >= 0.0)
    {
      return command_delay_sec_;
    }

    if (gripper_action_delay_sec_ > 0.0)
    {
      return gripper_action_delay_sec_;
    }

    if (!enable_gripper_control_)
    {
      return simulated_action_duration_sec_ > 0.0 ? simulated_action_duration_sec_ : 2.0;
    }

    return 0.0;
  }

  void scheduleSuccess(double leftFeedback, double rightFeedback, double delaySec = -1.0)
  {
    if (command_completed_ || success_pending_ || completion_timer_)
    {
      return;
    }

    success_pending_ = true;
    waiting_feedback_ = false;

    if (watchdog_timer_)
    {
      watchdog_timer_->cancel();
      watchdog_timer_.reset();
    }

    const double effectiveDelay = delaySec >= 0.0 ? delaySec : getEffectiveCompletionDelaySec();
    if (effectiveDelay <= 0.0)
    {
      markSuccess(leftFeedback, rightFeedback);
      return;
    }

    pending_left_feedback_ = leftFeedback;
    pending_right_feedback_ = rightFeedback;
    completion_timer_ = getNode()->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(effectiveDelay)),
      [this]()
      {
        if (command_completed_ || !success_pending_)
        {
          return;
        }
        markSuccess(pending_left_feedback_, pending_right_feedback_);
      });
  }

  bool isTargetReached(double feedbackValue) const
  {
    if (
      mode_ == je_software::msg::EndEffectorCommand::MODE_POSITION ||
      mode_ == je_software::msg::EndEffectorCommand::MODE_TORQUE)
    {
      return std::fabs(feedbackValue - position_) <= position_tolerance_;
    }
    // Preset mode currently has no reversible feedback mapping; any finite feedback is acceptable.
    return std::isfinite(feedbackValue);
  }

  static bool isValidTorqueCommand(const std::string & command)
  {
    return command == je_software::msg::EndEffectorCommand::CMD_OPEN ||
           command == je_software::msg::EndEffectorCommand::CMD_CLOSE;
  }

  static std::string normalizeCommand(std::string command)
  {
    std::transform(command.begin(), command.end(), command.begin(), ::tolower);
    return command;
  }

  void markSuccess(double leftFeedback, double rightFeedback)
  {
    if (command_completed_)
    {
      return;
    }

    command_completed_ = true;
    success_pending_ = false;
    waiting_feedback_ = false;

    if (watchdog_timer_)
    {
      watchdog_timer_->cancel();
      watchdog_timer_.reset();
    }
    if (completion_timer_)
    {
      completion_timer_->cancel();
      completion_timer_.reset();
    }

    RCLCPP_INFO(
      getLogger(),
      !enable_gripper_control_ ?
        "[CbCtrlGripper] Simulated gripper action success. left=%.4f right=%.4f mode=%d target_pos=%.4f preset=%d command=%s torque=%.4f" :
      wait_for_feedback_ ?
        "[CbCtrlGripper] Feedback reached target. left=%.4f right=%.4f mode=%d target_pos=%.4f preset=%d command=%s torque=%.4f" :
        "[CbCtrlGripper] Open-loop command duration elapsed. left=%.4f right=%.4f mode=%d target_pos=%.4f preset=%d command=%s torque=%.4f",
      leftFeedback, rightFeedback, mode_, position_, preset_, command_.c_str(), torque_);
    this->postSuccessEvent();
  }

  void markFailure(const char * reason)
  {
    if (command_completed_)
    {
      return;
    }

    command_completed_ = true;
    success_pending_ = false;
    waiting_feedback_ = false;

    if (watchdog_timer_)
    {
      watchdog_timer_->cancel();
      watchdog_timer_.reset();
    }
    if (completion_timer_)
    {
      completion_timer_->cancel();
      completion_timer_.reset();
    }

    RCLCPP_WARN(getLogger(), "[CbCtrlGripper] %s", reason);
    this->postFailureEvent();
  }

  int mode_{je_software::msg::EndEffectorCommand::MODE_POSITION};
  double position_{0.0};
  int preset_{0};
  bool left_valid_{true};
  bool right_valid_{false};
  std::string topic_{"/end_effector_cmd_lr"};
  double timeout_sec_{2.0};
  std::string feedback_topic_{"/joint_states_double_arm"};
  std::string legacy_feedback_topic_{"/oculus_init_joint_state"};
  double position_tolerance_{0.03};
  std::string command_{};
  double torque_{0.0};
  bool wait_for_feedback_{true};
  bool enable_gripper_control_{true};
  double simulated_action_duration_sec_{2.0};
  double gripper_action_delay_sec_{0.0};
  double command_delay_sec_{-1.0};

  bool waiting_feedback_{false};
  bool command_completed_{false};
  bool success_pending_{false};
  double pending_left_feedback_{0.0};
  double pending_right_feedback_{0.0};
  rclcpp::Time command_sent_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<je_software::msg::EndEffectorCommandLR>::SharedPtr publisher_;
  rclcpp::Subscription<common::msg::OculusInitJointState>::SharedPtr feedback_sub_;
  rclcpp::Subscription<common::msg::OculusInitJointState>::SharedPtr feedback_sub_legacy_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::TimerBase::SharedPtr completion_timer_;
};
}  // namespace cl_moveit2z
