"""
Confidential, Copyright 2025, Sony AI, All rights reserved

Internal functionality is thoroughly tested in gtest
Here we just test the python bindings and wheel
"""

import os
import pytest

from onnxruntime import SessionOptions

from ace_rt_profiles import ProfileManager, ProfileGuard, RTSubProfile
from ace_rt_profiles.utils import set_onnx_cpu_affinity

# pid of calling thread
SELF = 0


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    # reset env
    monkeypatch.delenv("ACE_LAB", raising=False)
    monkeypatch.delenv("DART_RUN_ID", raising=False)
    monkeypatch.setenv("CI", "true")

    # reset affinity
    os.sched_setaffinity(SELF, [i for i in range(os.cpu_count())])
    assert len(os.sched_getaffinity(SELF)) == os.cpu_count()

    # reset scheduler
    os.sched_setscheduler(SELF, os.SCHED_OTHER, os.sched_param(sched_priority=0))
    assert os.sched_getscheduler(SELF) == os.SCHED_OTHER
    assert os.sched_getparam(SELF).sched_priority == 0


@pytest.fixture
def manager():
    return ProfileManager.get_instance()


def test_profile_manager(manager):
    # test singleton is working properly
    manager_2 = ProfileManager.get_instance()
    assert manager is manager_2

    # test profiles are working properly
    assert manager.get_affinity("test_profile_manager") == [3, 4]
    manager.apply("test_profile_manager")
    assert os.sched_getaffinity(SELF) == {3, 4}
    assert os.sched_getscheduler(SELF) == os.SCHED_FIFO
    assert os.sched_getparam(SELF).sched_priority == 89


def test_profile_guard(manager):
    with ProfileGuard():
        # temporarily apply profile
        manager.apply("test_profile_manager")

        # confirm profile is applied
        assert os.sched_getaffinity(SELF) == {3, 4}
        assert os.sched_getscheduler(SELF) == os.SCHED_FIFO
        assert os.sched_getparam(SELF).sched_priority == 89

    # confirm profile is reverted
    assert len(os.sched_getaffinity(SELF)) == os.cpu_count()
    assert os.sched_getscheduler(SELF) == os.SCHED_OTHER
    assert os.sched_getparam(SELF).sched_priority == 0


def test_subprofile(manager):
    manager.apply(RTSubProfile(rt_profile="test_profile_manager", affinity_index=0))
    assert os.sched_getaffinity(SELF) == {3}
    assert os.sched_getscheduler(SELF) == os.SCHED_FIFO
    assert os.sched_getparam(SELF).sched_priority == 89

    manager.apply(RTSubProfile(rt_profile="test_profile_manager", affinity_index=1))
    assert os.sched_getaffinity(SELF) == {4}
    assert os.sched_getscheduler(SELF) == os.SCHED_FIFO
    assert os.sched_getparam(SELF).sched_priority == 89


def test_onnx():
    opts = SessionOptions()
    set_onnx_cpu_affinity("test_profile_manager", opts)
    assert opts.get_session_config_entry("session.intra_op_thread_affinities") == "4;5"
    assert opts.intra_op_num_threads == 3

    with pytest.raises(RuntimeError, match=r"intra_op_num_threads.* must be lower"):
        set_onnx_cpu_affinity("test_profile_manager_utils", opts)
