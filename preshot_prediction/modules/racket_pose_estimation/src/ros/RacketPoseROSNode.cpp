
// Confidential, Copyright 2025, Sony AI, All rights reserved.
#include "racket_pose_estimation/ros/RacketPoseROSNode.hpp"

#include <fstream>

#include "ace_interfaces/msg/logger_control.hpp"
#include "ace_interfaces/msg/player_detection.hpp"
#include "ace_interfaces/msg/player_pose2d.hpp"
#include "ace_interfaces/msg/racket_pose.hpp"
#include "ace_interfaces/msg/racket_pose_estimate.hpp"
#include "ace_loggers/ace_loggers.hpp"
#include "aps/SyncedImagesCallback.hpp"

namespace perception {
class RacketPoseROSNode::RacketPoseROSNodeImpl {
 public:
  RacketPoseROSNodeImpl() { sync_images_callback = std::make_shared<aps::SyncedImagesCallback>(); }

  void Start(std::shared_ptr<rclcpp::Node> ros_node, RacketPoseExtractor::SharedPtr racket_extractor) {
    extractor = racket_extractor;
    auto parameters = extractor->GetRacketParameters();
    this->ros_node = ros_node;

    rclcpp::QoS qos_persistent(rclcpp::KeepLast(1));
    qos_persistent.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_persistent.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    // Set up LoggerControl callback
    loggercontrol_sub = this->ros_node->create_subscription<ace_interfaces::msg::LoggerControl>(
      "/logger/control", qos_persistent,
      std::bind(&RacketPoseROSNodeImpl::LoggerControlCallback, this, std::placeholders::_1));
    sync_images_callback->Initialize(ros_node, extractor->GetCameraCalibrationParameters(), parameters->cameras_names,
                                     parameters->capture.camera_buffer_length, parameters->capture.worker_queue_length,
                                     parameters->capture.use_variable_response_time);

    // now set callback
    sync_images_callback->SetCallback(
      [this](const aps::SyncedImagesCallback::ImageBuffers::GroupsOfSyncedItems& group_of_items) {
        for (const auto& synced_items : group_of_items) {
          std::vector<RacketPoseExtractor::ImageData> images;
          for (const auto& it : synced_items.second) {
            const auto& msg = it.second;
            cv::Mat bayer_img;
            const void* data_ptr =
              static_cast<const void*>(msg->data.data() + msg->step * msg->offset_y + msg->offset_x);
            bayer_img = cv::Mat(msg->height, msg->width, CV_8UC1, const_cast<void*>(data_ptr), msg->step);
            images.emplace_back(std::pair(it.first, bayer_img));
          }
          extractor->OnImages(synced_items.first, images);
        }
      });

    const auto& racket_params = extractor->GetRacketParameters();

    rclcpp::QoS qos(rclcpp::KeepLast(5));
    for (auto& racket : racket_params->rackets) {
      std::stringstream topic_name;
      topic_name << "/sensors/racket" << racket.second.racket_id << "/pose";
      pose3d_pubs[racket.second.racket_id] =
        ros_node->create_publisher<ace_interfaces::msg::RacketPoseEstimate>(topic_name.str(), qos);

      for (size_t i = 0; i < racket.second.cameras.size(); ++i) {
        auto racket_id = racket.second.racket_id;
        auto& camera = racket.second.cameras[i];
        // if (camera.camera_calib_index == -1) {
        //   continue;
        // }
        topic_name = std::stringstream();
        topic_name << "/sensors/player" << racket.second.racket_id << "/" << camera.camera_name << "/player_pose_2d";
        if (player_detection_subs.find(topic_name.str()) != player_detection_subs.end()) {
          continue;
        }
        LOG(INFO) << racket.second.racket_id << "/" << camera.camera_calib_index
                  << " - Subscriping to player pose: " << topic_name.str();
        player_detection_subs[topic_name.str()] = ros_node->create_subscription<ace_interfaces::msg::PlayerPose2d>(
          topic_name.str(), 1, [&, racket_id, camera](ace_interfaces::msg::PlayerPose2d::SharedPtr msg) {
            Eigen::Vector4i roi;
            auto params = extractor->GetRacketParameters();
            const auto& camera_roi = params->rackets[racket_id].cameras[camera.camera_params_index];
            int w_size = params->extractor.roi.x();
            int h_size = params->extractor.roi.y();
            // construct the roi from the player's hand locations if possible (idx L: 9,R: 10)
            auto left_hand = msg->keypoints[9];
            auto right_hand = msg->keypoints[10];
            auto center =
              Eigen::Vector2i((left_hand.x * params->capture.left_hand + right_hand.x * params->capture.right_hand),
                              (left_hand.y * params->capture.left_hand + right_hand.y * params->capture.right_hand));
            roi.x() = std::max<int>(0, center.x() - w_size / 2);
            roi.y() = std::max<int>(0, center.y() - h_size / 2);
            roi.z() = w_size;
            roi.w() = h_size;

            roi = camera_roi.intersect_roi(roi);

            /*
            float inflation = 0.5;
            roi.x() = std::max<int>(0, msg->bbox[0] - msg->bbox[2] * inflation / 2);
            roi.y() = std::max<int>(0, msg->bbox[1] - msg->bbox[3] * inflation / 2);
            roi.z() = msg->bbox[2] + msg->bbox[2] * inflation;
            roi.w() = msg->bbox[3] + msg->bbox[2] * inflation;*/

            // std::cout << "Racket: " << racket.second.racket_id << " Camera: " << camera.camera_calib_index
            //           << "- Roi: " << roi.transpose() << std::endl;
            extractor->SetROI(racket_id, camera.camera_calib_index, roi);
          });
      }
    }

    extractor->SetRacketPose3DCallback([this](int seq_id, int racket_id, const RacketEstimatedPose::SharedPtr& pose) {
      const float iframe_rate = 1.0F / 1000.0F;
      const int sequence_number_to_ns = static_cast<int>(1e9F * iframe_rate);
      ace_interfaces::msg::RacketPoseEstimate msg;
      msg.pose.header.sequence_number = seq_id;
      msg.pose.header.stamp = rclcpp::Time(static_cast<int64_t>(static_cast<double>(seq_id) * sequence_number_to_ns));
      if (pose) {
        msg.pose.tracked = true;
        msg.pose.position.x = pose->position.x();
        msg.pose.position.y = pose->position.y();
        msg.pose.position.z = pose->position.z();
        msg.pose.orientation.x = pose->orientation.x();
        msg.pose.orientation.y = pose->orientation.y();
        msg.pose.orientation.z = pose->orientation.z();
        msg.pose.orientation.w = pose->orientation.w();
        msg.projection_error = pose->reprojection_err;
        msg.orientation_error = pose->orientation_error;
        msg.confidence = pose->orientation_confidence;
      } else {
        msg.pose.tracked = false;
        msg.projection_error = 1e9;
        msg.orientation_error = 1e9;
        msg.confidence = 0;
      }
      pose3d_pubs[static_cast<int>(racket_id)]->publish(msg);
    });
  }
  void Stop() {
    pose3d_pubs.clear();
    sync_images_callback->Stop();
    ros_exec = nullptr;
    ros_node = nullptr;
  }

  void Spin() {
    ros_exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    ros_thread = std::thread(std::bind(&RacketPoseROSNodeImpl::ROSWorkerThread, this));
    ros_exec->add_node(ros_node);
  }
  void ROSWorkerThread() { ros_exec->spin(); }

  void LoggerControlCallback(const ace_interfaces::msg::LoggerControl::ConstSharedPtr& msg_ptr) {
    const auto log_prefix = std::string(reinterpret_cast<const char*>(msg_ptr->log_prefix.data()));
    const auto log_name = ::datalogger::ConstructSubLogName(log_prefix, "racket_pose");
    extractor->SetDataLoggerFileName(log_name);
    LOG(INFO) << "Log filename got externally updated to:\n - \"" << log_prefix << "\"" << std::endl;
  }

  rclcpp::Subscription<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_sub;
  aps::SyncedImagesCallback::SharedPtr sync_images_callback;
  RacketPoseExtractor::SharedPtr extractor;

  std::thread ros_thread;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> ros_exec;
  std::shared_ptr<rclcpp::Node> ros_node;

  std::map<int, rclcpp::Publisher<ace_interfaces::msg::RacketPoseEstimate>::SharedPtr> pose3d_pubs;  // array of poses

  std::map<std::string, rclcpp::Subscription<ace_interfaces::msg::PlayerPose2d>::SharedPtr> player_detection_subs;
};

RacketPoseROSNode::RacketPoseROSNode() { impl_ = std::make_shared<RacketPoseROSNodeImpl>(); }

void RacketPoseROSNode::StartWithNode(std::shared_ptr<rclcpp::Node> node, RacketPoseExtractor::SharedPtr extractor) {
  impl_->Start(node, extractor);
}
void RacketPoseROSNode::Start(const std::string& node_name, RacketPoseExtractor::SharedPtr extractor) {
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(false);
  options.automatically_declare_parameters_from_overrides(false);
  options.start_parameter_event_publisher(false);
  options.start_parameter_services(false);
  // options.use_intra_process_comms(true);
  auto node = std::make_shared<rclcpp::Node>(node_name, options);
  impl_->Start(node, extractor);
  impl_->Spin();
}
void RacketPoseROSNode::Stop() { impl_->Stop(); }
}  // namespace perception
