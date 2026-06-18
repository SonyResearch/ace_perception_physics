// Confidential, Copyright 2025, Sony AI, All rights reserved
#include "player_pose/ros/PlayerPoseROSNode.hpp"

#include <cuda_common/TRTEngine.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "ace_interfaces/msg/logger_control.hpp"
#include "ace_interfaces/msg/player_detection.hpp"
#include "ace_interfaces/msg/player_pose.hpp"
#include "ace_interfaces/msg/player_pose2d.hpp"
#include "aps/SyncedImagesCallback.hpp"

namespace perception {
class PlayerPoseROSNode::PlayerPoseROSNodeImpl {
  class ImageMessagePacket {
   public:
    size_t sequence_number{0};
    std::vector<PlayerPoseExtractor::ImageData> images;
  } curr_message_packet;

 public:
  PlayerPoseROSNodeImpl() { sync_images_callback = std::make_shared<aps::SyncedImagesCallback>(); }

  void Start(std::shared_ptr<rclcpp::Node> ros_node, PlayerPoseExtractor::SharedPtr player_extractor,
             bool enable_bbox_pub, bool enable_pose2d_pub, bool enable_pose3d_pub) {
    this->enable_bbox_pub = enable_bbox_pub;
    this->enable_pose2d_pub = enable_pose2d_pub;
    this->enable_pose3d_pub = enable_pose3d_pub;
    this->extractor = player_extractor;
    auto parameters = extractor->GetPlayerParameters();
    this->ros_node = ros_node;

    sync_images_callback->Initialize(ros_node, extractor->GetCameraCalibrationParameters(), parameters->cameras_names,
                                     parameters->capture.camera_buffer_length, parameters->capture.worker_queue_length,
                                     parameters->capture.use_variable_response_time);

    rclcpp::QoS qos_persistent(rclcpp::KeepLast(1));
    qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    // Set up LoggerControl callback
    loggercontrol_sub = this->ros_node->create_subscription<ace_interfaces::msg::LoggerControl>(
      "/logger/control", qos_persistent,
      std::bind(&PlayerPoseROSNodeImpl::LoggerControlCallback, this, std::placeholders::_1));
    // now set callback
    sync_images_callback->SetCallback(
      [this](const aps::SyncedImagesCallback::ImageBuffers::GroupsOfSyncedItems& group_of_items) {
        for (const auto& synced_items : group_of_items) {
          std::vector<PlayerPoseExtractor::ImageData::SharedPtr> images;
          images.reserve(synced_items.second.size());
          for (const auto& it : synced_items.second) {
            const auto& camera_index = it.first;
            const auto& msg = it.second;

            auto data = std::make_shared<PlayerPoseExtractor::ImageData>();
            const void* data_ptr =
              static_cast<const void*>(msg->data.data() + msg->step * msg->offset_y + msg->offset_x);
            auto image = cv::Mat(msg->height, msg->width, CV_8UC1, const_cast<void*>(data_ptr), msg->step);

            data->camera_index = camera_index;
            data->image = std::move(image);
            images.emplace_back(data);
          }
          extractor->OnImages(synced_items.first, images);
        }
      });

    if (enable_pose2d_pub) {
      const auto& player_params = extractor->GetPlayerParameters();

      rclcpp::QoS qos(rclcpp::KeepLast(5));
      for (auto& player : player_params->players) {
        for (auto& camera : player.second.cameras) {
          std::stringstream topic_name;
          topic_name << "/sensors/player" << player.first << "/" << camera.second.camera_name << "/player_pose_2d";
          std::cout << "Adding: " << topic_name.str() << std::endl;

          auto key = PlayerPoseParameters::PlayerCameraPair(player.first, camera.second.camera_calib_index);
          pose2d_pubs[key] = ros_node->create_publisher<ace_interfaces::msg::PlayerPose2d>(topic_name.str(), qos);
        }
      }
      extractor->SetPlayerPose2DCallback([this, player_params](int seq_id, const PlayerPoseDetection::SharedPtr& pose) {
        const float iframe_rate = 1.0F / 1000.0F;
        const int sequence_number_to_ns = static_cast<int>(1e9F * iframe_rate);
        ace_interfaces::msg::PlayerPose2d msg;
        msg.header.sequence_number = seq_id;
        msg.header.stamp = rclcpp::Time(static_cast<int64_t>(static_cast<double>(seq_id) * sequence_number_to_ns));
        msg.bbox[0] = pose->bbox.x();
        msg.bbox[1] = pose->bbox.y();
        msg.bbox[2] = pose->bbox.z();
        msg.bbox[3] = pose->bbox.w();
        for (size_t i = 0; i < pose->keypoints.size(); ++i) {
          msg.keypoints[i].x = pose->keypoints[i].x();
          msg.keypoints[i].y = pose->keypoints[i].y();
          msg.keypoints[i].z = pose->keypoints[i].z();
        }
        auto key = PlayerPoseParameters::PlayerCameraPair(pose->player_index, pose->camera_index);
        pose2d_pubs[key]->publish(msg);
      });
    }
    if (enable_pose3d_pub) {
      const auto& player_params = extractor->GetPlayerParameters();

      rclcpp::QoS qos(rclcpp::KeepLast(5));
      for (auto& player : player_params->players) {
        std::stringstream topic_name;
        topic_name << "/sensors/player" << player.second.player_id << "/pose";
        pose3d_pubs[player.second.player_id] =
          ros_node->create_publisher<ace_interfaces::msg::PlayerPose>(topic_name.str(), qos);
      }

      extractor->SetPlayerPose3DCallback([this](int seq_id, int player_id, const PlayerPoseEstimate::SharedPtr& pose) {
        const float iframe_rate = 1.0F / 1000.0F;
        const int sequence_number_to_ns = static_cast<int>(1e9F * iframe_rate);
        ace_interfaces::msg::PlayerPose msg;
        msg.header.sequence_number = seq_id;
        msg.header.stamp = rclcpp::Time(static_cast<int64_t>(static_cast<double>(seq_id) * sequence_number_to_ns));
        if (pose) {
          for (size_t i = 0; i < pose->keypoints.size(); ++i) {
            msg.keypoints[i].x = pose->keypoints[i].x();
            msg.keypoints[i].y = pose->keypoints[i].y();
            msg.keypoints[i].z = pose->keypoints[i].z();

            msg.projection_error[i] = pose->projection_error[i];
            msg.confidences[i] = pose->confidences[i];
          }
        } else {
          for (size_t i = 0; i < 17; ++i) {
            msg.projection_error[i] = 1e3;
            msg.confidences[i] = 0;
          }
        }
        pose3d_pubs[player_id]->publish(msg);
      });
    }
  }
  void Stop() {
    pose2d_pubs.clear();
    pose3d_pubs.clear();

    sync_images_callback->Stop();
    ros_exec = nullptr;
    ros_node = nullptr;
  }

  void Spin() {
    ros_exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    ros_thread = std::thread(std::bind(&PlayerPoseROSNodeImpl::ROSWorkerThread, this));
    ros_exec->add_node(ros_node);
  }
  void ROSWorkerThread() { ros_exec->spin(); }

  void LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr) {
    const auto log_prefix = std::string(reinterpret_cast<const char*>(msg_ptr->log_prefix.data()));
    const auto log_name = ::datalogger::ConstructSubLogName(log_prefix, "player_pose");
    extractor->SetDataLoggerFileName(log_name);
    LOG(INFO) << "Log filename got externally updated to:\n - \"" << log_prefix << "\"" << std::endl;
  }

  rclcpp::Subscription<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_sub;
  aps::SyncedImagesCallback::SharedPtr sync_images_callback;
  PlayerPoseExtractor::SharedPtr extractor;
  bool enable_bbox_pub{false};
  bool enable_pose2d_pub{false};
  bool enable_pose3d_pub{false};

  std::thread ros_thread;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> ros_exec;
  std::shared_ptr<rclcpp::Node> ros_node;

  std::map<PlayerPoseParameters::PlayerCameraPair, rclcpp::Publisher<ace_interfaces::msg::PlayerPose2d>::SharedPtr>
    pose2d_pubs;                                                                             // array for cameras
  std::map<int, rclcpp::Publisher<ace_interfaces::msg::PlayerPose>::SharedPtr> pose3d_pubs;  // array of players
};

PlayerPoseROSNode::PlayerPoseROSNode() { impl_ = std::make_shared<PlayerPoseROSNodeImpl>(); }

void PlayerPoseROSNode::StartWithNode(std::shared_ptr<rclcpp::Node> node, PlayerPoseExtractor::SharedPtr extractor,
                                      bool enable_bbox_pub, bool enable_pose2d_pub, bool enable_pose3d_pub) {
  impl_->Start(node, extractor, enable_bbox_pub, enable_pose2d_pub, enable_pose3d_pub);
}
void PlayerPoseROSNode::Start(const std::string& node_name, PlayerPoseExtractor::SharedPtr extractor,
                              bool enable_bbox_pub, bool enable_pose2d_pub, bool enable_pose3d_pub) {
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(false);
  options.automatically_declare_parameters_from_overrides(false);
  options.start_parameter_event_publisher(false);
  options.start_parameter_services(false);
  // options.use_intra_process_comms(true);
  auto node = std::make_shared<rclcpp::Node>(node_name, options);
  impl_->Start(node, extractor, enable_bbox_pub, enable_pose2d_pub, enable_pose3d_pub);
  impl_->Spin();
}
void PlayerPoseROSNode::Stop() { impl_->Stop(); }
}  // namespace perception
