# Confidential, Copyright 2025, Sony AI, All rights reserved.
"""Helper script to plot errors for racket pose estimation vs ground truth.
Make sure to have `racket_gt_{index}.csv` and `racket_est_{index}.csv` available when running this script """

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def parse_file(file):
    """Parse csv file"""
    with open(file, encoding="utf8") as fp:  # pylint: disable=invalid-name
        reader_obj = csv.reader(fp, delimiter="\t")
        values = np.array(list(reader_obj)).astype(float)

        return {int(row[0]): row[1:] for row in values}


def plot(axis: plt.Axes, x, data, axis_index, label, color, std=None, err_label=""):  # pylint: disable=invalid-name
    """Plot errors"""
    y = np.array([d[axis_index - 1] for d in data])  # pylint: disable=invalid-name
    x = np.array(list(range(len(y))))
    mean = np.mean(y)
    axis.set_ylabel(label)
    if isinstance(color, np.ndarray):
        scatter = axis.scatter(x, y, label=label, c=color, cmap="nipy_spectral", s=3)
        cbar = plt.colorbar(scatter, orientation="vertical")
        cbar.ax.set_ylabel(err_label, rotation=90)
    else:
        axis.plot(x, y, label=label, color=color)
    axis.axhline(mean, linestyle="--")
    print(f"{label} - Mean: {mean}±{np.std(y)}")

    if std is not None:
        std = np.array([d[axis_index - 1] for d in std])
        axis.plot(x, y - std, linestyle="--", color="orange")
        axis.plot(x, y + std, linestyle="--", color="orange")
    axis.legend()


def main(window_size=100, index=0, pos=True, use_norm=True):  # pylint: disable=too-many-statements
    """Main entry point"""
    gt = parse_file(f"racket_gt_{index}.csv")  # pylint: disable=invalid-name
    est = parse_file(f"racket_est_{index}.csv")  # pylint: disable=invalid-name

    max_reproj = 5
    max_ori_err = 0.05
    min_conf = 0.4

    est = {e: est[e] for e in est if est[e][7] < max_reproj and est[e][8] < max_ori_err and est[e][9] > min_conf}

    err = []
    err_std = []
    sequences = [d for d in est if d in gt]

    def round_value(x):  # pylint: disable=invalid-name
        """Round values to -180,+180 range"""
        while x > 90:
            x -= 180
        while x < -90:
            x += 180
        return x

    ball_offset = 0.105
    for i in range(len(sequences) - window_size):
        idx_window = sequences[i : i + window_size]
        window = []
        for idx in idx_window:
            row = []
            # calculate error for position
            value = 0
            for j in range(3):
                value += (est[idx][j] - gt[idx][j]) ** 2
            value = value**0.5
            if value > 0.5:
                value = ball_offset
            row.append(value - ball_offset)
            row.append(0)
            row.append(0)
            est_rot = R.from_quat(est[idx][3:7])
            gt_rot = R.from_quat(gt[idx][3:7])
            rot_diff = est_rot * gt_rot.inv()
            if use_norm:
                rot_diff = rot_diff.magnitude() * 180 / np.pi  # degrees=True)
                row.append(round_value(np.linalg.norm(rot_diff)))
                row.append(0)
                row.append(0)
            else:
                rot_diff = rot_diff.as_euler("xyz", degrees=True)
                for j in range(3):
                    row.append(round_value(rot_diff[j]))

            row.append(est[idx][-3])
            row.append(est[idx][-2])
            row.append(est[idx][-1])
            window.append(row)
        vec = np.mean(window, axis=0)
        err.append(vec)

        vec = np.std(window, axis=0)
        err_std.append(vec)

    err = np.array(err)
    err_std = np.array(err_std)
    np.set_printoptions(suppress=True)
    fig = plt.figure()

    if window_size == 1:
        err_std = None
    if pos:
        err_idx = 6
        err = np.array(err)
        # colors=np.array(err[:,err_idx]*100/np.max(err[:,err_idx]),dtype=int)
        axis = fig.add_subplot(3, 1, 1)

        plot(axis, sequences, err * 100, 1, "err.pos (mm)", "b", err_std, err_label="reprojection error %")
        axis = fig.add_subplot(3, 1, 2)
        plot(axis, sequences, err, 7, "reprojection error", "b", err_std)
        axis = fig.add_subplot(3, 1, 3)
        plot(axis, sequences, err, 9, "confidence", "b", err_std)
    else:
        err_idx = 7
        err = np.array(err)
        colors = np.array(err[:, err_idx] * 100 / np.max(err[:, err_idx]), dtype=int)
        if use_norm:
            axis = fig.add_subplot(3, 1, 1)
            plot(axis, sequences, err, 4, "err.rot (deg)", colors, err_std, err_label="orientation error %")
            axis = fig.add_subplot(3, 1, 2)
            plot(axis, sequences, err, 8, "orientation error", "b", err_std)
            axis = fig.add_subplot(3, 1, 3)
            plot(axis, sequences, err, 9, "confidence", "b", err_std)
        else:
            axis = fig.add_subplot(3, 2, 1)
            plot(axis, sequences, err, 4, "err.rx", colors, err_std)
            axis = fig.add_subplot(3, 2, 3)
            plot(axis, sequences, err, 5, "err.ry", colors, err_std)
            axis = fig.add_subplot(3, 2, 5)
            plot(axis, sequences, err, 6, "err.rz", colors, err_std)

            axis = fig.add_subplot(3, 2, 2)
            plot(axis, sequences, err, 8, "orientation error", "b", err_std)
            # axis = fig.add_subplot(3, 2, 4)
            # plot(axis, sequences, err, 7, "reprojection error", "b", err_std)
            # axis = fig.add_subplot(3, 2, 6)
            # plot(axis, sequences, err, 9, "confidence", "b", err_std)

    plt.show()


main(window_size=1, index=2, pos=True, use_norm=True)
