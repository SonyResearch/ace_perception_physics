# Camera Parameters #

The camera parameters are defined in [camera_parameters.hpp](../include/aps/camera_parameters.hpp).
They are grouped according to the Technical Reference of the camera (FLIR Blackfly S U3-16S2).
An archived version of the Technical Reference with a detailed description of the parameters can be found [here](https://github.com/SonyResearch/project_ace_literature/blob/master/files/BFS-U3-16S2-Technical-Reference.pdf) (revised 22.11.2017).

The parameters are stored under `parameters/camera_parameters/` in yaml files.

## Acquisition control ##
* acquisition_mode [-]
* exposure_mode [-]
* exposure_time [us]
* exposure_auto [-]
* acquisition_frame_rate [Hz]
* acquisition_frame_rate_enable [-]

## Analog control ##
* gain [dB]
* gain_auto [-]
* black_level [%]
* black_level_clamping_enable [-]
* balance_ratio_blue [-]
* balance_ratio_red [-]
* balance_white_auto [-]
* gamma [-]
* gamma_enable [-]

## Image format control ##
* width [px]
* height [px]
* offset_x [px]
* offset_y [px]
* pixel_format [-]
* adc_bit_depth [-]

## Chunk data control ##
* chunk_mode_active [-]
* chunk_exposure_time [-]

## Stream buffer control ##
* manual_stream_buffer_count [-]
* stream_buffer_count_mode [-]
* stream_buffer_handling_mode [-]
*
