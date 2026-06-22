# Confidential, Copyright 2024, Sony AI, All rights reserved.
"""Provide some visualization & UI functionalities for calibration scripts."""

import functools
import logging
import select
import sys

import matplotlib
import numpy
from matplotlib import pyplot
from mpl_toolkits import mplot3d  # noqa: F401  # pylint: disable = unused-import

log = logging.getLogger(__name__)


def configure():
    """Prepare visualization environment."""
    # Disable logs that we do not care about
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def initialize():
    """Initialize visualization environment."""
    pyplot.ion()
    matplotlib.rcParams["figure.figsize"] = 19, 14


def _toggle_plot_visibility_callback(fig, plot_lines_dict, event):
    legend_line = event.artist
    plot_lines = plot_lines_dict[legend_line]
    is_visible = not plot_lines[0].get_visible()
    for plot_line in plot_lines:
        plot_line.set_visible(is_visible)
    legend_line.set_alpha(1.0 if is_visible else 0.2)
    fig.canvas.draw()


def draw_pose(
    axis,
    pose,
    length=0.15,
    label=None,
    label_color="k",
    label_offset=None,
):
    """Draws XYZ-axes basis arrows (colored as RGB, respectively)."""
    translation_vect = pose[:3, 3]
    handles = []
    if label_offset is None:
        label_offset = [0.01, 0.01, 0.01]

    for dir_vect, color in zip(numpy.hsplit(pose[:3, :3], [1, 2]), numpy.hsplit(numpy.eye(3), [1, 2])):
        arrow_end_vect, color = translation_vect + length * dir_vect.ravel(), color.ravel()
        handles.append(
            axis.plot(
                [translation_vect[0], arrow_end_vect[0]],
                [translation_vect[1], arrow_end_vect[1]],
                [translation_vect[2], arrow_end_vect[2]],
                color=color,
            )[0]
        )
    if label is not None:
        t_offset = translation_vect + label_offset
        handles.append(
            axis.text(
                t_offset[0],
                t_offset[1],
                t_offset[2],
                label,
                size=10,
                zorder=1,
                color=label_color,
            )
        )
    return handles


def generate_3d_plot(points_list, labels, poses=None, save_as=None, show_fig=True, title=None):
    """
    Generate a 3D plot of the observed trajectories for the robot.

    Optionally, it also visualizes the basis vectors of the robot and each camera.
    """
    if not show_fig and save_as is None:
        return None, None

    fig = pyplot.figure(title)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    plot_lines = []
    for points_3d, label in zip(points_list, labels):
        points_3d = points_3d.reshape((-1, 3))
        plot_lines.append(
            axis.plot(
                points_3d[:, 0],
                points_3d[:, 1],
                points_3d[:, 2],
                label=label,
            )[0]
        )
    legend_lines = axis.legend().get_texts()
    plot_lines_dict = {}
    for legend_line, plot_line in zip(legend_lines, plot_lines):
        legend_line.set_picker(True)
        plot_lines_dict[legend_line] = [plot_line]
    if poses is not None:
        for legend_line, pose, label in zip(legend_lines, poses, labels):
            plot_lines_dict[legend_line].extend(draw_pose(axis, pose, label=label))
    pyplot.tight_layout()
    pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1, wspace=0.1)
    if save_as is not None:
        pyplot.savefig(save_as)
    if show_fig:
        fig.canvas.mpl_connect(
            "pick_event",
            functools.partial(_toggle_plot_visibility_callback, fig, plot_lines_dict),
        )
        mng = pyplot.get_current_fig_manager()
        if hasattr(mng.window, "showMaximized"):
            mng.window.showMaximized()
        elif hasattr(mng.window, "maxsize"):
            mng.resize(*mng.window.maxsize())
        fig.canvas.flush_events()
    else:
        pyplot.close(fig)

    return fig, axis


def generate_signals_plot(time_stamps_list, points_list, labels, save_as=None, show_fig=True, title=None):
    """Generate a 2D plot of the robot trajectory as reported by the robot and as observed by each camera."""
    if not show_fig and save_as is None:
        return None, None

    fig, axes = pyplot.subplots(3, 1, sharex=True)
    for axis, y_label in zip(axes, ["x", "y", "z"]):
        axis.set_ylabel(y_label)
    plot_lines = []
    for time_stamps, points_3d, label in zip(time_stamps_list, points_list, labels):
        points_3d = points_3d.reshape((-1, 3))
        valid_label_mask = numpy.isfinite(points_3d[:, 2])
        valid_time_stamps = time_stamps[valid_label_mask]
        valid_points_3d = points_3d[valid_label_mask]

        for axis_index, axis in enumerate(axes):
            plot_lines.append(
                axis.scatter(
                    valid_time_stamps,
                    valid_points_3d[:, axis_index],
                    s=1,
                    label=label,
                )
            )
    legend_lines = []
    for axis_index, axis in enumerate(axes):
        legend_lines.append(axis.legend().get_texts())
    legend_lines = numpy.array(legend_lines).T.ravel().tolist()
    plot_lines_dict = {}
    for legend_line, plot_line in zip(legend_lines, plot_lines):
        legend_line.set_picker(True)
        plot_lines_dict[legend_line] = [plot_line]
    pyplot.tight_layout()
    pyplot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.1, wspace=0.1)
    if save_as is not None:
        pyplot.savefig(save_as)
    if show_fig:
        fig.canvas.manager.set_window_title(title)
        fig.canvas.mpl_connect(
            "pick_event",
            functools.partial(_toggle_plot_visibility_callback, fig, plot_lines_dict),
        )
        mng = pyplot.get_current_fig_manager()
        if hasattr(mng.window, "showMaximized"):
            mng.window.showMaximized()
        else:
            mng.resize(*mng.window.maxsize())
        fig.canvas.flush_events()
    else:
        pyplot.close(fig)

    return fig, axes


def wait_for_user_input():
    """Keep the interactive plots until the user press Enter key."""
    print("Press Enter or close all figures to exit...")
    while len(pyplot.get_fignums()):
        if select.select([sys.stdin], [], [], 0.01)[0]:
            break
        pyplot.gcf().canvas.flush_events()
