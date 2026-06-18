# Blackfly S Camera#

The default APS camera for project Ace is the FLIR Blackfly S USB3 16S2C ([BFS-U3-16S2C-CS](https://www.flir.co.uk/products/blackfly-s-usb3/?model=BFS-U3-16S2C-CS)). A summary of its specifications is listed below and the full datasheet can be found [here](https://github.com/SonyResearch/project_ace_literature/blob/master/files/BFS-U3-16S2-Datasheet.pdf).

## Spinnaker SDK ##
The camera can be controlled using FLIR's Spinnaker SDK. The most frequently used functions (and parameters) are implemented in [blackfly_s_camera.hpp](../include/aps/blackfly_s_camera.hpp), which also allows for conveniently setting the parameters defined in [camera_parameters.hpp](../include/aps/camera_parameters.hpp).

## Specifications ##
* ADC: 10-bit, 12-bit
* Chroma: Color
* Frame Rate: 226
* Lens Mount: CS-mount
* Megapixels: 1.6
* Part Number: BFS-U3-16S2C-CS
* Pixel Size: 3.45
* Readout Method: Global shutter
* Resolution: 1440 × 1080
* Sensor Format: 1/2.9"
* Sensor Name: Sony IMX273
* Sensor Type: CMOS
