#!/usr/bin/env python3
# type: ignore
# SPDX-License-Identifier: MIT
"""Event subscriber wrapper, synch. to APS clock, and publishing of corrected timestamps."""

import time

import ace_yaml as yaml
import rclpy
from ament_index_python.packages import get_package_share_directory
from evs import DisplayPubTCEvents, ZeroCopyEventSubscriber  # pylint: disable = no-name-in-module
from rclpy.qos import QoSProfile
from tqdm import tqdm

from ace_interfaces.msg import BatchTCEvents


class RosEventSubscriber:
    """ROS subscriber for event and clock synchronization parameters.

    Subscribes to events and KF parameters for clock synchronization
    parameters, corrects timestamps accordingly and publishes the corrected events.

    """

    def __init__(self):
        """Initialize all parameters."""
        init = False
        self.subscription = ZeroCopyEventSubscriber.ZeroCopyEventSubscriber(init)

        # Publish time corrected events
        rclpy.init()
        self.node = rclpy.create_node("pub_events_corrected")
        qos_profile = QoSProfile(depth=10)
        self.publisher = self.node.create_publisher(BatchTCEvents, "events_corrected", qos_profile)

        # Get index for camera subscription from launch file to chose topic subscription
        cam_index = self.node.declare_parameter("cam_index", "1")
        cam_index = self.node.get_parameter("cam_index").get_parameter_value().string_value

        # Handles for topic subscriptions
        self.handles = []
        topic_string = "/events" + cam_index
        self.handles.append(self.subscription.add_event_subscription(topic_string))
        self.handles.append(self.subscription.add_clock_subscription("/clock_params"))

        time.sleep(1)

    def __iter__(self):
        """Iterate, then execute __next__."""
        return self

    def __next__(self) -> BatchTCEvents():
        """Get zero-copy event and clock-kf message from cpp and correct timestamp."""
        event_struc = BatchTCEvents()
        # Get event message
        (
            ret,
            event_struc.number_timestamps,
            event_struc.number_events,
            event_struc.t,
            event_struc.x,
            event_struc.y,
            event_struc.p,
        ) = self.subscription.get_latest_event_message(self.handles[0])
        if ret:
            self.subscription.release_latest_event_message(self.handles[0])
        # Get clock parameters
        (
            ret_clock,
            offset,
            skew,
            ts_trigger,
        ) = self.subscription.get_latest_clock_message(self.handles[1])
        if ret_clock:
            self.subscription.release_latest_clock_message(self.handles[1])
            # Correct timestamps
            for i in range(event_struc.number_timestamps):
                event_struc.t[i] = (event_struc.t[i] - offset) * skew + ts_trigger
        # Publish time corrected events
        self.publisher.publish(event_struc)
        return event_struc


def visualize(visualizer_cpp, events_to_visualize):
    """Visualizes events with C++ DisplayPubTCEvents class."""
    ev_accumulated = 0
    if events_to_visualize.number_timestamps > 0:
        for i in range(events_to_visualize.number_timestamps):
            ev_accumulated += events_to_visualize.number_events[i]
        array_x = events_to_visualize.x[0:ev_accumulated]
        array_y = events_to_visualize.y[0:ev_accumulated]
        array_p = events_to_visualize.p[0:ev_accumulated]
        visualizer_cpp.draw_pixels_py(ev_accumulated, array_x, array_y, array_p)


if __name__ == "__main__":
    # Read parameters for visualization and publishing
    # TODO(Raphaela): use parameter class # pylint: disable=fixme, line-too-long
    PARAM_PATH = get_package_share_directory("evs") + "/parameters/evs_settings.yaml"
    with open(PARAM_PATH, "r", encoding="utf-8") as params_file:
        data_params = yaml.load(params_file, Loader=yaml.FullLoader)
    height_cam = data_params["height"]
    width_cam = data_params["width"]
    visualize_param = data_params["visualize_subscriber"]

    ros_event_subscriber = RosEventSubscriber()
    event_loader = tqdm(ros_event_subscriber)

    # Add the visualizer
    visualizer = DisplayPubTCEvents.DisplayPubTCEvents()
    display = visualizer.setup_display([width_cam, height_cam])

    while rclpy.ok():
        for events in event_loader:
            if visualize_param:
                visualize(visualizer, events)
    ros_event_subscriber.node.destroy_node()
    rclpy.shutdown()
