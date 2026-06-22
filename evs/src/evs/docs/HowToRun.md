First, source the ROS environment: "source install/setup.bash"

Source the the openeb environment with: "/openeb/build/utils/scripts/setup_env.sh"

Source fast_dds settings: "source tools/setup_fast_dds.bash"

Next, compile the ROS packages:

colcon build --symlink-install --packages-select base_parameters ace_interfaces evs
Make the subscriber executable with

"chmod +x src/sensors/evs/event_subscriber/event_subscriber.py"

Make sure you defined the correct parameters in

/project_ace/sensors/evs/parameters/evs_settings.yaml .

Now you can run the code: ros2 launch evs evs_setup.launch.py

The data recordings and bias files are saved in /project_ace. out_m.raw is the recording of the master camera. Depending on how many slave cameras there are out_s(i).raw is the recording of the i_th slave camera. The .bias file extension holds the bias settings.
