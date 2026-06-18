// Confidential, Copyright 2024, Sony AI, All rights reserved.
#include "aps/camera_parameters.hpp"

#include "ace_loggers/ace_loggers.hpp"

namespace aps {

CameraParameters::CameraParameters() : BaseParameters("camera_parameters") {}

bool CameraParameters::UpdateParametersFromYaml() {
  try {
    // acquisition control
    acquisition_mode =
      static_cast<Spinnaker::AcquisitionModeEnums>(yaml_node_["acquisition_control"]["acquisition_mode"].as<uint>());
    exposure_mode =
      static_cast<Spinnaker::ExposureModeEnums>(yaml_node_["acquisition_control"]["exposure_mode"].as<uint>());
    exposure_time = yaml_node_["acquisition_control"]["exposure_time"].as<double>();
    exposure_auto =
      static_cast<Spinnaker::ExposureAutoEnums>(yaml_node_["acquisition_control"]["exposure_auto"].as<uint>());
    acquisition_frame_rate = yaml_node_["acquisition_control"]["acquisition_frame_rate"].as<double>();
    acquisition_frame_rate_enable = yaml_node_["acquisition_control"]["acquisition_frame_rate_enable"].as<bool>();

    // analog control
    gain = yaml_node_["analog_control"]["gain"].as<double>();
    gain_auto = static_cast<Spinnaker::GainAutoEnums>(yaml_node_["analog_control"]["gain_auto"].as<uint>());
    black_level = yaml_node_["analog_control"]["black_level"].as<double>();
    black_level_clamping_enable = yaml_node_["analog_control"]["black_level_clamping_enable"].as<bool>();
    balance_ratio_blue = yaml_node_["analog_control"]["balance_ratio_blue"].as<double>();
    balance_ratio_red = yaml_node_["analog_control"]["balance_ratio_red"].as<double>();
    balance_white_auto =
      static_cast<Spinnaker::BalanceWhiteAutoEnums>(yaml_node_["analog_control"]["balance_white_auto"].as<uint>());
    gamma = yaml_node_["analog_control"]["gamma"].as<double>();
    gamma_enable = yaml_node_["analog_control"]["gamma_enable"].as<bool>();

    // image format control
    width = yaml_node_["image_format_control"]["width"].as<int>();
    height = yaml_node_["image_format_control"]["height"].as<int>();
    offset_x = yaml_node_["image_format_control"]["offset_x"].as<int>();
    offset_y = yaml_node_["image_format_control"]["offset_y"].as<int>();
    pixel_format =
      static_cast<Spinnaker::PixelFormatEnums>(yaml_node_["image_format_control"]["pixel_format"].as<uint>());
    adc_bit_depth =
      static_cast<Spinnaker::AdcBitDepthEnums>(yaml_node_["image_format_control"]["adc_bit_depth"].as<uint>());

    // chunk data control
    chunk_mode_active = yaml_node_["chunk_data_control"]["chunk_mode_active"].as<bool>();
    chunk_exposure_time = yaml_node_["chunk_data_control"]["exposure_time"].as<bool>();
    chunk_timestamp = yaml_node_["chunk_data_control"]["timestamp"].as<bool>();

    // stream buffer control
    manual_stream_buffer_count = yaml_node_["stream_buffer_control"]["manual_stream_buffer_count"].as<int>();
    stream_buffer_count_mode = static_cast<Spinnaker::StreamBufferCountModeEnum>(
      yaml_node_["stream_buffer_control"]["stream_buffer_count_mode"].as<uint>());
    stream_buffer_handling_mode = static_cast<Spinnaker::StreamBufferHandlingModeEnum>(
      yaml_node_["stream_buffer_control"]["stream_buffer_handling_mode"].as<uint>());
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

bool CameraParameters::UpdateYamlFromParameters() {
  try {
    // acquisition control
    yaml_node_["acquisition_control"]["acquisition_mode"] = static_cast<uint>(acquisition_mode);
    yaml_node_["acquisition_control"]["exposure_mode"] = static_cast<uint>(exposure_mode);
    yaml_node_["acquisition_control"]["exposure_time"] = exposure_time;
    yaml_node_["acquisition_control"]["exposure_auto"] = static_cast<uint>(exposure_auto);
    yaml_node_["acquisition_control"]["acquisition_frame_rate"] = acquisition_frame_rate;
    yaml_node_["acquisition_control"]["acquisition_frame_rate_enable"] = acquisition_frame_rate_enable;

    // analog control
    yaml_node_["analog_control"]["gain"] = gain;
    yaml_node_["analog_control"]["gain_auto"] = static_cast<uint>(gain_auto);
    yaml_node_["analog_control"]["black_level"] = black_level;
    yaml_node_["analog_control"]["black_level_clamping_enable"] = black_level_clamping_enable;
    yaml_node_["analog_control"]["balance_ratio_blue"] = balance_ratio_blue;
    yaml_node_["analog_control"]["balance_ratio_red"] = balance_ratio_red;
    yaml_node_["analog_control"]["balance_white_auto"] = static_cast<uint>(balance_white_auto);
    yaml_node_["analog_control"]["gamma"] = gamma;
    yaml_node_["analog_control"]["gamma_enable"] = gamma_enable;

    // image format control
    yaml_node_["image_format_control"]["width"] = width;
    yaml_node_["image_format_control"]["height"] = height;
    yaml_node_["image_format_control"]["offset_x"] = offset_x;
    yaml_node_["image_format_control"]["offset_y"] = offset_y;
    yaml_node_["image_format_control"]["pixel_format"] = static_cast<uint>(pixel_format);
    yaml_node_["image_format_control"]["adc_bit_depth"] = static_cast<uint>(adc_bit_depth);

    // chunk data control
    yaml_node_["chunk_data_control"]["chunk_mode_active"] = chunk_mode_active;
    yaml_node_["chunk_data_control"]["exposure_time"] = chunk_exposure_time;
    yaml_node_["chunk_data_control"]["timestamp"] = chunk_timestamp;

    // stream buffer control
    yaml_node_["stream_buffer_control"]["manual_stream_buffer_count"] = manual_stream_buffer_count;
    yaml_node_["stream_buffer_control"]["stream_buffer_count_mode"] = static_cast<uint>(stream_buffer_count_mode);
    yaml_node_["stream_buffer_control"]["stream_buffer_handling_mode"] = static_cast<uint>(stream_buffer_handling_mode);
  } catch (YAML::Exception &e) {
    LOG(ERROR) << e.what() << "\n";
    return false;
  }

  return true;
}

}  // namespace aps
