// Confidential, Copyright 2024, Sony AI, All rights reserved.
#pragma once

#include <Spinnaker.h>

#include "base_parameters/base_parameters.hpp"

namespace aps {

class CameraParameters final : public BaseParameters {
 public:
  using SharedPtr = std::shared_ptr<CameraParameters>;
  using ConstSharedPtr = std::shared_ptr<const CameraParameters>;

  explicit CameraParameters();

  // The parameters are grouped according to the Technical Reference of the
  // camera (FLIR Blackfly S U3-16S2). An archived version of the Technical
  // Reference with a detailed description of the parameters can be found here:
  // https://github.com/SonyResearch/project_ace_literature/blob/master/files/BFS-U3-16S2-Technical-Reference.pdf
  // (revised 22.11.2017).
  // Acquisition control.
  Spinnaker::AcquisitionModeEnums acquisition_mode;  // [-]
  Spinnaker::ExposureModeEnums exposure_mode;        // [-]
  double exposure_time;                              // [us]
  Spinnaker::ExposureAutoEnums exposure_auto;        // [-]
  double acquisition_frame_rate;                     // [Hz]
  bool acquisition_frame_rate_enable;                // [-]

  // Analog control.
  double gain;                                          // [dB]
  Spinnaker::GainAutoEnums gain_auto;                   // [-]
  double black_level;                                   // [%]
  bool black_level_clamping_enable;                     // [-]
  double balance_ratio_blue;                            // [-]
  double balance_ratio_red;                             // [-]
  Spinnaker::BalanceWhiteAutoEnums balance_white_auto;  // [-]
  double gamma;                                         // [-]
  bool gamma_enable;                                    // [-]

  // Image format control.
  int width;                                  // [px]
  int height;                                 // [px]
  int offset_x;                               // [px]
  int offset_y;                               // [px]
  Spinnaker::PixelFormatEnums pixel_format;   // [-]
  Spinnaker::AdcBitDepthEnums adc_bit_depth;  // [-]

  // Chunk data control.
  bool chunk_mode_active;    // [-]
  bool chunk_exposure_time;  // [-]
  bool chunk_timestamp;      // [-]

  // Stream buffer control.
  int manual_stream_buffer_count;                                       // [-]
  Spinnaker::StreamBufferCountModeEnum stream_buffer_count_mode;        // [-]
  Spinnaker::StreamBufferHandlingModeEnum stream_buffer_handling_mode;  // [-]

 private:
  bool UpdateParametersFromYaml() override;
  bool UpdateYamlFromParameters() override;
};

}  // namespace aps
