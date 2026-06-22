**Multi camera EVS data recording using ROS nodes and APS – EVS clock synchronization**

With this program you can run multiple EVS cameras by launching a single python script. Retrieve the camera’s serial number by calling ./metavision_hal_ls’ in the ‘./openeb/build/bin’ folder after plugging the camera. You can now set serial numbers of the the master and slave cameras in /project_ace/src/sensors/evs/parameters/evs_settings.yaml. Alternatively, you can set the path to a .raw recording that you want t.o replay. Important: When setting the serial number use a “_” in front of the number, e.g. _00001870. Make sure to also indicate the number of cameras and ‘from_file’ either true or false, depending on whether you want to read from a file or use a plugged-in camera.

Set the ‘aps_freq’ to the framerate of the APS cameras and their exposure time in microseconds. This is required for clock synchronization, as two ‘special events’ in the EVS system are triggered that occur with APS framerate and exposure time.

Next, you can set the ROS spin rate (in microseconds) to publish event messages at a certain rate. Event messages are in the form of “BatchTCEvents.msg”: number_events is a vector that holds the amount of events at each timestamp. t is a vector with the timestamps. x is a vector that holds vectors of the x-pixel events that occur at each timestamp. The same holds for y (y-pixel) and p (polarity).

Another ROS message that is published at the given spin rate is ClockParamsKF.msg . It holds the newest clock offset, skew and the last time a ‘special event’ occurred (in APS time).

In the evs_settings.yaml, you can also indicate if you would like to visualize the events in the subscriber (after timestamp correction).

The ‘height’ and ‘width’ parameters (in pixels) are needed for the visualization and differ for Gen3 and Gen4 cameras.

You can also set the path to a ‘bias_file’ in order to set the biases of the EVS camera as defined in the file.

The code will create as many .raw recordings as there are cameras or data recordings to replay from. The output data is automatically stored as “out_m.raw” for the master camera and “out_s(slave index).raw” for slave cameras.

For more information how to set up the EVS camera system, refer to: <a href="https://sonyai.atlassian.net/wiki/spaces/SON/pages/2593718312/Guide+for+EVS+cameras+set-up" target="_blank">EVS camera system</a>
