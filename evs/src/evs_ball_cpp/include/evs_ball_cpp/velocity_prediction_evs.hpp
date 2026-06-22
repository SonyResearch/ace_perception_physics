// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "evs_ball_interfaces/msg/velocities_image.hpp"
#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_event_rate_noise_filter_module.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/facilities/i_geometry.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_trigger_in.h"
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "rclcpp/rclcpp.hpp"
#include "evs_ball_cpp/frame_generation.hpp"
#include "evs_ball_cpp/trt_engine.hpp"
#include "evs_ball_cpp/velocity_evs_parameters.hpp"

namespace velocity_prediction_evs {

class VelocityPredictionEVS : public rclcpp::Node {
 public:
  explicit VelocityPredictionEVS(const rclcpp::NodeOptions &options);
  ~VelocityPredictionEVS() override;

 private:
  std::unordered_map<std::string, std::unique_ptr<Metavision::Device>> cams_;
  std::atomic<bool> canceled_;
  int64_t num_bytes_master_;
  int64_t num_bytes_slave_0_;
  int64_t num_bytes_slave_1_;

  std::vector<std::vector<cv::cuda::HostMem>> event_frames_;

  bool complete_;
  int current_seq_number_;

  bool InitializeCamera(std::string, bool is_master);

  void StartRecording();
  void StopRecording();

  VelocityEVSParameters params_;
  std::vector<std::string> camera_names_evs_;

  std::vector<rclcpp::Publisher<evs_ball_interfaces::msg::VelocitiesImage>::SharedPtr> pubs_;

  void WorkerThreadFrameGenerationSlave();
  void WorkerThreadFrameGenerationSlave1();
  void WorkerThreadFrameGenerationMaster();
  std::thread t_generation_slave_;
  std::thread t_generation_slave_1_;
  std::thread t_generation_master_;

  void PublishRosMessage(std::unordered_map<std::string, float> ball_position);
  void WorkerThreadDetectionMaster();
  void WorkerThreadDetectionSlave();
  void WorkerThreadDetectionSlave1();
  std::thread t_detection_master_;
  std::thread t_detection_slave_;
  std::thread t_detection_slave_1_;
  std::unique_ptr<TensorRTEngine> trt_detection_ptr_master_;
  std::unique_ptr<TensorRTEngine> trt_detection_ptr_slave0_;
  std::unique_ptr<TensorRTEngine> trt_detection_ptr_slave1_;

  cudaStream_t main_streams_[12];  // use as many streams as needed
  int stream_idx_;
  std::vector<std::unique_ptr<FrameGeneration>> processor_;

  std::vector<std::queue<int>> index_queues_;

  Metavision::I_EventsStream *event_stream_ptr_master_;
  Metavision::I_EventsStream *event_stream_ptr_slave_;
  Metavision::I_EventsStream *event_stream_ptr_slave_1_;
  Metavision::I_EventsStreamDecoder *decoder_ptr_master_;
  Metavision::I_EventsStreamDecoder *decoder_ptr_slave_;
  Metavision::I_EventsStreamDecoder *decoder_ptr_slave_1_;

  // A ball position vector for every camera
  std::vector<std::unordered_map<std::string, float>> ball_positions_master_;
  std::vector<std::unordered_map<std::string, float>> ball_positions_slave_;
  std::vector<std::unordered_map<std::string, float>> ball_positions_slave1_;
  std::vector<std::unordered_map<std::string, float>> ball_positions_slave2_;

  // Definition for trigger only

  int64 sp_ts_master_;
  int64 sp_ts_slave_;
  std::queue<int64> sp_ts_queue_master_;
  std::queue<int64> sp_ts_queue_slave_;
  std::queue<int64> sp_ts_queue_slave_1_;

  double sequence_number_to_ns_;
  int old_seq_;
};

}  // namespace evs_ball_cpp
