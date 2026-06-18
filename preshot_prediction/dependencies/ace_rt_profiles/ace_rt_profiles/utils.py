"""Confidential, Copyright 2025, Sony AI, All rights reserved"""

from ace_rt_profiles import ProfileManager

import colored_glog as glog
from onnxruntime import SessionOptions

MAX_INTRA_OP_NUM_THREADS = 8


def set_onnx_cpu_affinity(profile: str, opts: SessionOptions) -> None:
    """
    Apply the cpu affinity of the given real time profile to the given onnx session

    Args:
        profile: Real time profile
        opts: Session options to apply cpu affinity to

    Raises:
        RuntimeError: If number of threads is greater than or equal to max allowed value
    """
    # onxx uses 1-index affinities, so we add +1 to the cpus listed in the profile
    cpu_affinity = ProfileManager.get_instance().get_affinity(profile)
    affinity_mask = ";".join([str(cpu + 1) for cpu in cpu_affinity])

    # number of affinities must be equal to intra_op_num_threads - 1
    # because onnx does not set affinity on the main thread
    opts.intra_op_num_threads = len(cpu_affinity) + 1
    opts.add_session_config_entry("session.intra_op_thread_affinities", affinity_mask)

    if opts.intra_op_num_threads >= MAX_INTRA_OP_NUM_THREADS:
        raise RuntimeError(
            f"intra_op_num_threads={opts.intra_op_num_threads} must be lower than {MAX_INTRA_OP_NUM_THREADS}. "
            f"Real time profile: {profile}, onnx affinity mask: {affinity_mask}"
        )

    glog.info(f"[INFO] Setting CPU affinity mask for ONNX runtime session to: {affinity_mask}")
