// Confidential, Copyright 2024, Sony AI, All rights reserved.

#include <glog/logging.h>
#include "evs_ball_cpp/velocity_prediction_evs.hpp"

namespace velocity_prediction_evs {

VelocityPredictionEVS::VelocityPredictionEVS(const rclcpp::NodeOptions& options)
  : rclcpp::Node("velocity_prediction_evs", options) {
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging(this->get_name());
  }

  const std::string& params_path = declare_parameter<std::string>("params_path");

  params_.Initialize(params_path);
  params_.PrintParameters();
  sequence_number_to_ns_ = 1e9 / 1000;
  camera_names_evs_.push_back(params_.camera_names[0]);
  LOG(INFO) << "camera name: " << params_.camera_names[0];
  camera_names_evs_.push_back(params_.camera_names[1]);
  LOG(INFO) << "camera name: " << params_.camera_names[1];
  camera_names_evs_.push_back(params_.camera_names[2]);
  LOG(INFO) << "camera name: " << params_.camera_names[2];
  canceled_.store(false);
  stream_idx_ = 0;
  sp_ts_master_ = 0;
  current_seq_number_ = 0;
  old_seq_ = 0;

  // Topics
  std::string topic_master =
    std::string(this->get_namespace()) + "/evs" + params_.master_serial_number + "/" + "ball_detection";
  pubs_.emplace_back(this->create_publisher<evs_ball_interfaces::msg::VelocitiesImage>(topic_master, 1));
  if (params_.number_cameras > 1) {
    std::string topic_slave =
      std::string(this->get_namespace()) + "/evs" + params_.slave_serial_number_0 + "/" + "ball_detection";
    pubs_.emplace_back(this->create_publisher<evs_ball_interfaces::msg::VelocitiesImage>(topic_slave, 1));
    std::string topic_slave_1 =
      std::string(this->get_namespace()) + "/evs" + params_.slave_serial_number_1 + "/" + "ball_detection";
    pubs_.emplace_back(this->create_publisher<evs_ball_interfaces::msg::VelocitiesImage>(topic_slave_1, 1));
  }

  if (params_.number_cameras > 1) {
    InitializeCamera(params_.slave_serial_number_0, false);
    usleep(5000);
    InitializeCamera(params_.slave_serial_number_1, false);
  }
  usleep(50000);  // Sleep to ensure all slave cameras are ready, then start master camera.
  InitializeCamera(params_.master_serial_number, true);

  // External trigger control

  for (int i = 0; i < params_.parts; i++) {  // arbitrary number of instances that are allocated in memory
    std::vector<cv::cuda::HostMem> tmp_frame_vector;
    for (int i = 0; i < params_.number_cameras; i++) {
      cv::cuda::HostMem tmp_frame(cv::cuda::HostMem::PAGE_LOCKED);
      tmp_frame.create(params_.height, params_.width, CV_16FC2);
      tmp_frame_vector.push_back(tmp_frame);
    }
    event_frames_.push_back(tmp_frame_vector);
  }
  // Variable initialization for the different cameras
  for (int i = 0; i < params_.number_cameras; i++) {
    index_queues_.emplace_back(std::queue<int>());
  }

  // streams
  for (auto& main_stream : main_streams_) {
    cudaStreamCreate(&main_stream);
  }

  auto* trigger_in_ptr_master = cams_[params_.master_serial_number]->get_facility<Metavision::I_TriggerIn>();
  trigger_in_ptr_master->enable(Metavision::I_TriggerIn::Channel::Main);

  auto* trigger_in_decoder_ptr_master =
    cams_[params_.master_serial_number]->get_facility<Metavision::I_EventDecoder<Metavision::EventExtTrigger>>();
  trigger_in_decoder_ptr_master->add_event_buffer_callback(
    [=](const Metavision::EventExtTrigger* begin_sp_master, const Metavision::EventExtTrigger* end_sp_master) {
      for (const auto* ev_sp_master = begin_sp_master; ev_sp_master != end_sp_master; ++ev_sp_master) {
        if (ev_sp_master->t - sp_ts_master_ >
            (1000000.0 / params_.trigger_freq) - (params_.trigger_exposure_time + 1000.0)) {
          sp_ts_master_ = ev_sp_master->t;
          sp_ts_queue_master_.push(sp_ts_master_);
          sp_ts_queue_slave_.push(sp_ts_master_);
          sp_ts_queue_slave_1_.push(sp_ts_master_);
        }
      }
    });

  processor_.push_back(std::make_unique<FrameGeneration>(&event_frames_, &index_queues_[0], params_.dt_accumulate_us, 0,
                                                         params_.height, params_.width, params_.channels, params_.parts,
                                                         &sp_ts_queue_master_, params_.trigger_freq));

  processor_.push_back(std::make_unique<FrameGeneration>(&event_frames_, &index_queues_[1], params_.dt_accumulate_us, 1,
                                                         params_.height, params_.width, params_.channels, params_.parts,
                                                         &sp_ts_queue_slave_, params_.trigger_freq));

  processor_.push_back(std::make_unique<FrameGeneration>(&event_frames_, &index_queues_[2], params_.dt_accumulate_us, 2,
                                                         params_.height, params_.width, params_.channels, params_.parts,
                                                         &sp_ts_queue_slave_1_, params_.trigger_freq));

  auto* cddecoder_ptr_master =
    cams_[params_.master_serial_number]->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

  cddecoder_ptr_master->add_event_buffer_callback(
    [=](const Metavision::EventCD* begin_master, const Metavision::EventCD* end_master) {
      processor_[0]->Process(begin_master, end_master);
    });

  if (params_.number_cameras > 1) {
    auto* cddecoder_ptr_slave =
      cams_[params_.slave_serial_number_0]->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    cddecoder_ptr_slave->add_event_buffer_callback(
      [=](const Metavision::EventCD* begin_slave, const Metavision::EventCD* end_slave) {
        processor_[1]->Process(begin_slave, end_slave);
      });
    LOG(INFO) << "Callback slave0 added.";

    auto* cddecoder_ptr_slave_1 =
      cams_[params_.slave_serial_number_1]->get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();
    cddecoder_ptr_slave_1->add_event_buffer_callback(
      [=](const Metavision::EventCD* begin_slave_1, const Metavision::EventCD* end_slave_1) {
        processor_[2]->Process(begin_slave_1, end_slave_1);
      });
    LOG(INFO) << "Callback slave1 added.";
  }

  std::cout << "Callbacks added.\n";
  LOG(INFO) << "Num cams: " << params_.number_cameras;

  trt_detection_ptr_master_ = std::make_unique<TensorRTEngine>(
    params_.model_path, params_.batch_size, params_.number_streams, params_.number_cameras, params_.width,
    params_.height, params_.num_bbox, params_.threshold, params_.velocity_constant);

  std::cout << "Starting metavision recording.\n";
  StartRecording();

  t_generation_master_ = std::thread(std::bind(&VelocityPredictionEVS::WorkerThreadFrameGenerationMaster, this));
  t_generation_master_.detach();

  if (params_.number_cameras > 1) {
    t_generation_slave_ = std::thread(std::bind(&VelocityPredictionEVS::WorkerThreadFrameGenerationSlave, this));
    t_generation_slave_.detach();
    t_generation_slave_1_ = std::thread(std::bind(&VelocityPredictionEVS::WorkerThreadFrameGenerationSlave1, this));
    t_generation_slave_1_.detach();
  }

  t_detection_master_ = std::thread(std::bind(&VelocityPredictionEVS::WorkerThreadDetectionMaster, this));
  t_detection_master_.detach();
}

void VelocityPredictionEVS::WorkerThreadFrameGenerationMaster() {
  LOG(INFO) << "Starting frame generation thread: " << params_.master_serial_number << " master camera.\n";

  decoder_ptr_master_ = cams_[params_.master_serial_number]->get_facility<Metavision::I_EventsStreamDecoder>();

  while (!canceled_.load()) {
    event_stream_ptr_master_->poll_buffer();
    uint8_t* raw_data_master = event_stream_ptr_master_->get_latest_raw_data(num_bytes_master_);
    decoder_ptr_master_->decode(raw_data_master, raw_data_master + num_bytes_master_);
  }
}

void VelocityPredictionEVS::WorkerThreadFrameGenerationSlave() {
  LOG(INFO) << "Starting frame generation thread: " << params_.slave_serial_number_0 << " slave camera.\n";

  decoder_ptr_slave_ = cams_[params_.slave_serial_number_0]->get_facility<Metavision::I_EventsStreamDecoder>();

  while (!canceled_.load()) {
    event_stream_ptr_slave_->poll_buffer();
    uint8_t* raw_data_slave = event_stream_ptr_slave_->get_latest_raw_data(num_bytes_slave_0_);
    decoder_ptr_slave_->decode(raw_data_slave, raw_data_slave + num_bytes_slave_0_);
  }
}
void VelocityPredictionEVS::WorkerThreadFrameGenerationSlave1() {
  LOG(INFO) << "Starting frame generation thread: " << params_.slave_serial_number_1 << " slave camera.\n";

  decoder_ptr_slave_1_ = cams_[params_.slave_serial_number_1]->get_facility<Metavision::I_EventsStreamDecoder>();

  while (!canceled_.load()) {
    event_stream_ptr_slave_1_->poll_buffer();
    uint8_t* raw_data_slave_1 = event_stream_ptr_slave_1_->get_latest_raw_data(num_bytes_slave_1_);
    decoder_ptr_slave_1_->decode(raw_data_slave_1, raw_data_slave_1 + num_bytes_slave_1_);
  }
}

void VelocityPredictionEVS::PublishRosMessage(std::unordered_map<std::string, float> ball_position) {
  evs_ball_interfaces::msg::VelocitiesImage message_out;
  message_out.header.sequence_number = current_seq_number_;

  message_out.header.stamp = rclcpp::Time(static_cast<int64_t>(current_seq_number_ * sequence_number_to_ns_));

  message_out.velocities.emplace_back();

  // Fill up message with predictions
  message_out.velocities.back().center.x = ball_position["u"];
  message_out.velocities.back().center.y = ball_position["v"];
  message_out.velocities.back().radius = ball_position["r"];
  message_out.velocities.back().vel_y = ball_position["vel_y"];
  message_out.velocities.back().vel_x = ball_position["vel_x"];

  // Fill up message with unceertainties
  message_out.velocities.back().x_sigma = ball_position["u_sigma"];
  message_out.velocities.back().y_sigma = ball_position["v_sigma"];
  message_out.velocities.back().r_sigma = ball_position["r_sigma"];
  message_out.velocities.back().vel_y_sigma = ball_position["vel_y_sigma"];
  message_out.velocities.back().vel_x_sigma = ball_position["vel_x_sigma"];

  // Publish message to ROS
  pubs_[static_cast<uint64>(ball_position["idx"])]->publish(std::move(message_out));
}

// Prediction threads for each camera
void VelocityPredictionEVS::WorkerThreadDetectionMaster() {
  LOG(INFO) << "Starting prediction thread: " << params_.master_serial_number << " master camera.\n";
  while (!canceled_.load()) {
    bool complete = (!index_queues_[0].empty()) && (!index_queues_[1].empty()) && (!index_queues_[2].empty());

    if (complete) {
      int smallest = 0;
      if ((index_queues_[0].front() <= index_queues_[1].front()) &&
          (index_queues_[0].front() <= index_queues_[2].front())) {
        smallest = index_queues_[0].front();
      } else if ((index_queues_[1].front() <= index_queues_[2].front()) &&
                 (index_queues_[1].front() <= index_queues_[0].front())) {
        smallest = index_queues_[1].front();
      } else if ((index_queues_[2].front() <= index_queues_[0].front()) &&
                 (index_queues_[2].front() <= index_queues_[1].front())) {
        smallest = index_queues_[2].front();
      }
      if (smallest > old_seq_) {
        old_seq_ = smallest;

        trt_detection_ptr_master_->Predict(event_frames_[(smallest % params_.parts)], stream_idx_, main_streams_,
                                           ball_positions_master_);
        current_seq_number_ = smallest;

        for (auto& ball_position : ball_positions_master_) {
          PublishRosMessage(ball_position);
        }

        stream_idx_ = (stream_idx_ + 1) % 3;
        ball_positions_master_.clear();
      }
      if (smallest == index_queues_[0].front()) {
        index_queues_[0].pop();
      }
      if (smallest == index_queues_[1].front()) {
        index_queues_[1].pop();
      }
      if (smallest == index_queues_[2].front()) {
        index_queues_[2].pop();
      }
    }
  }
}

VelocityPredictionEVS::~VelocityPredictionEVS() {
  // Stop recording.
  LOG(INFO) << "Stopping metavision recording.\n";
  StopRecording();
}

bool VelocityPredictionEVS::InitializeCamera(std::string serial_number, bool is_master) {
  try {
    cams_[serial_number] = Metavision::DeviceDiscovery::open(serial_number);
  } catch (Metavision::HalException& e) {
    std::cout << e.what() << std::endl;
    return false;
  }

  auto* device_control_ptr = cams_[serial_number]->get_facility<Metavision::I_CameraSynchronization>();
  auto* biases_ptr = cams_[serial_number]->get_facility<Metavision::I_LL_Biases>();

  // Apply camera parameters.
  biases_ptr->set("bias_diff", params_.bias_diff);
  biases_ptr->set("bias_diff_off", params_.bias_diff_off);
  biases_ptr->set("bias_diff_on", params_.bias_diff_on);
  biases_ptr->set("bias_hpf", params_.bias_hpf);
  biases_ptr->set("bias_refr", params_.bias_refr);

  // Set master/slave mode.
  if (is_master) {
    device_control_ptr->set_mode_master();
  } else {
    device_control_ptr->set_mode_slave();
  }

  // Set ERC
  cams_[serial_number]->get_facility<Metavision::I_ErcModule>()->enable(true);
  cams_[serial_number]->get_facility<Metavision::I_ErcModule>()->set_cd_event_rate(params_.erc_event_rate);

  return true;
}

void VelocityPredictionEVS::StartRecording() {
  // Start all slave cameras
  if (params_.number_cameras > 1) {
    event_stream_ptr_slave_ = cams_[params_.slave_serial_number_0]->get_facility<Metavision::I_EventsStream>();
    event_stream_ptr_slave_->start();
    LOG(INFO) << "started slave camera: " << params_.slave_serial_number_0;

    event_stream_ptr_slave_1_ = cams_[params_.slave_serial_number_1]->get_facility<Metavision::I_EventsStream>();
    event_stream_ptr_slave_1_->start();
    LOG(INFO) << "started slave camera: " << params_.slave_serial_number_1;
  }

  // Start master camera.
  usleep(10000);
  event_stream_ptr_master_ = cams_[params_.master_serial_number]->get_facility<Metavision::I_EventsStream>();

  event_stream_ptr_master_->start();

  LOG(INFO) << "started master camera: " << params_.master_serial_number;
}

void VelocityPredictionEVS::StopRecording() {
  canceled_.store(true);

  if (t_generation_master_.joinable()) {
    t_generation_master_.join();
  }

  if (params_.number_cameras > 1) {
    if (t_generation_slave_.joinable()) {
      t_generation_slave_.join();
    }
    if (t_generation_slave_1_.joinable()) {
      t_generation_slave_1_.join();
    }
  }

  if (t_detection_master_.joinable()) {
    t_detection_master_.join();
  }

  // Stop master camera.
  event_stream_ptr_master_->stop();

  // Stop all slave cameras.
  if (params_.number_cameras > 1) {
    event_stream_ptr_slave_->stop();
    event_stream_ptr_slave_1_->stop();
  }
}
}  // namespace velocity_prediction_evs
