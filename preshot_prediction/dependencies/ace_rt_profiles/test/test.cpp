// Confidential, Copyright 2025, Sony AI, All rights reserved.

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sched.h>

#include <cstdlib>
#include <thread>

#include "ace_rt_profiles/profile_guard.hpp"
#include "ace_rt_profiles/profile_manager.hpp"

namespace ace_rt_profiles {

// helper that is friends with ProfileManager class
// allows creating and destroying managers at will
// useful for unit testing independently of singleton
class ProfileManagerTester {
 public:
  ProfileManager manager;
};

class TestProfileManager : public testing::Test {
 protected:
  // reset settings before every test
  void SetUp() override {
    // reset env
    setenv("ACE_LAB", "test", 1);
    unsetenv("ACE_PC");
    unsetenv("CI");
    unsetenv("DART_RUN_ID");

    // reset affinity
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
      CPU_SET(i, &cpu_set);
    }
    ASSERT_EQ(sched_setaffinity(kSelf, sizeof(cpu_set), &cpu_set), EXIT_SUCCESS);

    // reset policy
    const sched_param param = {.sched_priority = 0};
    ASSERT_EQ(sched_setscheduler(kSelf, SCHED_OTHER, &param), EXIT_SUCCESS);

    ASSERT_NO_FATAL_FAILURE(AssertDefault());
  }

  // assert the current affinity matches the expected one
  static void AssertAffinity(const std::vector<int>& cores) {
    cpu_set_t cpu_set;
    ASSERT_EQ(sched_getaffinity(kSelf, sizeof(cpu_set), &cpu_set), EXIT_SUCCESS);

    if (cores.empty()) {
      ASSERT_EQ(CPU_COUNT(&cpu_set), std::thread::hardware_concurrency());
    } else {
      cpu_set_t expect;
      CPU_ZERO(&expect);

      for (const int core : cores) {
        CPU_SET(core, &expect);
      }

      ASSERT_TRUE(CPU_EQUAL(&cpu_set, &expect));
    }
  }

  // assert the current policy matches the expected one
  static void AssertPolicy(int expect_policy, int expect_priority) {
    sched_param param;
    ASSERT_EQ(sched_getscheduler(kSelf), expect_policy);
    ASSERT_EQ(sched_getparam(kSelf, &param), EXIT_SUCCESS);
    ASSERT_EQ(param.sched_priority, expect_priority);
  }

  // assert if settings are set to default values
  static void AssertDefault() {
    ASSERT_NO_FATAL_FAILURE(AssertAffinity({}));
    ASSERT_NO_FATAL_FAILURE(AssertPolicy(SCHED_OTHER, 0));
  }

  // --- ATTRS ---

  // pid of calling thread
  static constexpr int kSelf = 0;
};

TEST_F(TestProfileManager, NoValidLab) {
  // env vars are not set
  unsetenv("ACE_LAB");
  unsetenv("ACE_PC");
  EXPECT_THROW(ProfileManagerTester(), std::runtime_error);

  // lab is set, but not pc
  setenv("ACE_LAB", "foo", 1);
  EXPECT_THROW(ProfileManagerTester(), std::runtime_error);

  // pc is set, but not lab
  unsetenv("ACE_LAB");
  setenv("ACE_PC", "bar", 1);
  EXPECT_THROW(ProfileManagerTester(), std::runtime_error);

  // lab and pc are set, but file does not exist
  setenv("ACE_LAB", "foo", 1);
  EXPECT_THROW(ProfileManagerTester(), std::runtime_error);

  // expect no changes
  EXPECT_NO_FATAL_FAILURE(AssertDefault());
}

TEST_F(TestProfileManager, MissingProfile) {
  // no issues expected when loading the config file
  setenv("ACE_PC", "base", 1);
  const ProfileManagerTester tester;

  // but trying to apply a profile that doesn't exist throws
  EXPECT_THROW(tester.manager.Apply("foo"), std::runtime_error);

  // expect no changes
  EXPECT_NO_FATAL_FAILURE(AssertDefault());
}

TEST_F(TestProfileManager, Invalid) {
  setenv("ACE_PC", "invalid", 1);
  const ProfileManagerTester tester;

  EXPECT_THROW(tester.manager.Apply(-1, "wrong_pid"), std::system_error);
  EXPECT_THROW(tester.manager.Apply("wrong_cpu"), std::system_error);
  EXPECT_THROW(tester.manager.Apply("wrong_policy"), std::runtime_error);
  EXPECT_THROW(tester.manager.Apply("missing_priority"), std::runtime_error);
  EXPECT_THROW(tester.manager.Apply("affinity_not_list"), std::runtime_error);
  EXPECT_THROW(tester.manager.Apply("policy_not_string"), std::runtime_error);
  EXPECT_THROW(tester.manager.Apply("priority_not_int"), std::runtime_error);
  EXPECT_THROW(tester.manager.Apply("affinity_negative"), std::runtime_error);

  EXPECT_NO_FATAL_FAILURE(AssertDefault());
}

TEST_F(TestProfileManager, CIAndDart) {
  setenv("ACE_PC", "base", 1);
  setenv("CI", "true", 1);
  setenv("DART_RUN_ID", "true", 1);

  // DART should be prioritized over CI and ACE_LAB
  // Since profiles 'A' and 'test_profile_manager' don't exist in dart.yaml, they should throw
  const ProfileManagerTester tester1;
  EXPECT_THROW(tester1.manager.Apply("A"), std::runtime_error);
  EXPECT_THROW(tester1.manager.Apply("test_profile_manager"), std::runtime_error);
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  // DART_RUN_ID is not set, so CI should be prioritized over ACE_LAB
  // Since profiles 'A' and 'test_profile_manager' exist in test/base.yaml and ci.yaml respectively
  // 'A' should throw and 'test_profile_manager' should work
  unsetenv("DART_RUN_ID");
  const ProfileManagerTester tester2;
  EXPECT_THROW(tester2.manager.Apply("A"), std::runtime_error);
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  tester2.manager.Apply("test_profile_manager");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({3, 4}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 89));
}

TEST_F(TestProfileManager, Base) {
  setenv("ACE_PC", "base", 1);
  const ProfileManagerTester tester;

  tester.manager.Apply("A");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({0}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 89));

  tester.manager.Apply("B");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({1}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_OTHER, 0));

  tester.manager.Apply("C");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({2}));
}

TEST_F(TestProfileManager, Override) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  tester.manager.Apply("A");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({0}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 89));

  tester.manager.Apply("B");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({1}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_RR, 91));

  tester.manager.Apply("C");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({3, 4}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
}

TEST_F(TestProfileManager, Discard) {
  setenv("ACE_PC", "discard", 1);
  const ProfileManagerTester tester;

  // --- expect A unchanged ---
  tester.manager.Apply("A");
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  // --- expect B unchanged ---
  tester.manager.Apply("B");
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  // --- expect C unchanged ---
  tester.manager.Apply("C");
  EXPECT_NO_FATAL_FAILURE(AssertDefault());
}

TEST_F(TestProfileManager, GetAffinity) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  EXPECT_THAT(tester.manager.GetAffinity("C"), testing::ElementsAre(3, 4));
  EXPECT_THROW(tester.manager.GetAffinity("foo"), std::runtime_error);
}

TEST_F(TestProfileManager, MakeThreads) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  std::vector<RTThread> threads = tester.manager.MakeThreads("C");
  EXPECT_EQ(threads.size(), 2);

  for (size_t i = 0; i < threads.size(); ++i) {
    threads.at(i).Run([i]() {
      EXPECT_NO_FATAL_FAILURE(AssertAffinity({i == 0 ? 3 : 4}));
      EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
    });
  }

  for (auto& thread : threads) {
    if (thread.Joinable()) {
      thread.Join();
    }
  }

  setenv("ACE_PC", "discard", 1);
  const ProfileManagerTester tester2;
  EXPECT_TRUE(tester2.manager.MakeThreads("B").empty());
  EXPECT_TRUE(tester2.manager.MakeThreads("C").empty());
}

TEST_F(TestProfileManager, MakeThreadsDefault) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;
  EXPECT_THROW(tester.manager.MakeThreads("A", 123, static_cast<AffinityMode>(123)), std::runtime_error);
}

TEST_F(TestProfileManager, MakeThreadsSequential) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  std::vector<RTThread> threads = tester.manager.MakeThreads("C", 3, AffinityMode::kSequential);
  EXPECT_EQ(threads.size(), 3);

  for (size_t i = 0; i < threads.size(); ++i) {
    threads.at(i).Run([i]() {
      EXPECT_NO_FATAL_FAILURE(AssertAffinity({i == 1 ? 4 : 3}));
      EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
    });
  }

  for (auto& thread : threads) {
    if (thread.Joinable()) {
      thread.Join();
    }
  }
}

TEST_F(TestProfileManager, MakeThreadsHomogeneous) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  std::vector<RTThread> threads = tester.manager.MakeThreads("C", 3, AffinityMode::kHomogeneous);
  EXPECT_EQ(threads.size(), 3);

  for (auto& thread : threads) {
    thread.Run([]() {
      EXPECT_NO_FATAL_FAILURE(AssertAffinity({3, 4}));
      EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
    });
  }

  for (auto& thread : threads) {
    if (thread.Joinable()) {
      thread.Join();
    }
  }
}

TEST_F(TestProfileManager, MakeThreadsHybrid) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  // --- hybrid == sequential ---

  std::vector<RTThread> threads = tester.manager.MakeThreads("C", 2, AffinityMode::kHybrid);
  EXPECT_EQ(threads.size(), 2);

  for (size_t i = 0; i < threads.size(); ++i) {
    threads.at(i).Run([i]() {
      EXPECT_NO_FATAL_FAILURE(AssertAffinity({i == 0 ? 3 : 4}));
      EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
    });
  }

  for (auto& thread : threads) {
    if (thread.Joinable()) {
      thread.Join();
    }
  }

  // --- hybrid == homogeneous ---

  threads = tester.manager.MakeThreads("C", 3, AffinityMode::kHybrid);
  EXPECT_EQ(threads.size(), 3);

  for (auto& thread : threads) {
    thread.Run([]() {
      EXPECT_NO_FATAL_FAILURE(AssertAffinity({3, 4}));
      EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
    });
  }

  for (auto& thread : threads) {
    if (thread.Joinable()) {
      thread.Join();
    }
  }
}

TEST_F(TestProfileManager, SubProfile) {
  setenv("ACE_PC", "override", 1);
  const ProfileManagerTester tester;

  // missing profile
  EXPECT_THROW(tester.manager.Apply(RTSubProfile{.rt_profile = "foo", .affinity_index = 0}), std::runtime_error);
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  // invalid index
  EXPECT_THROW(tester.manager.Apply(RTSubProfile{.rt_profile = "C", .affinity_index = 99}), std::out_of_range);
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  tester.manager.Apply(RTSubProfile{.rt_profile = "C", .affinity_index = 0});
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({3}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));

  tester.manager.Apply(RTSubProfile{.rt_profile = "C", .affinity_index = 1});
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({4}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));

  tester.manager.Apply("C");
  EXPECT_NO_FATAL_FAILURE(AssertAffinity({3, 4}));
  EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 10));
}

// simple test for the singleton functionality
TEST_F(TestProfileManager, Singleton) {
  setenv("ACE_PC", "base", 1);
  const ProfileManager& manager1 = ProfileManager::GetInstance();
  manager1.Apply("A");

  // even after changing env var, singleton does not re-initialize
  // and still points to the same reference
  setenv("ACE_LAB", "foo", 1);
  const ProfileManager& manager2 = ProfileManager::GetInstance();
  manager2.Apply("A");
  EXPECT_EQ(&manager1, &manager2);
}

TEST_F(TestProfileManager, Guard) {
  setenv("ACE_PC", "base", 1);
  const ProfileManagerTester tester;

  {
    const ProfileGuard guard;
  }

  // confirm nothing changed
  EXPECT_NO_FATAL_FAILURE(AssertDefault());

  {
    // temporarily apply profile A
    const ProfileGuard guard_a;
    tester.manager.Apply("A");
    EXPECT_NO_FATAL_FAILURE(AssertAffinity({0}));
    EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 89));

    {
      // temporarily apply profile B
      const ProfileGuard guard_b;
      tester.manager.Apply("B");
      EXPECT_NO_FATAL_FAILURE(AssertAffinity({1}));
      EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_OTHER, 0));
    }

    // confirm profile B is reverted
    EXPECT_NO_FATAL_FAILURE(AssertAffinity({0}));
    EXPECT_NO_FATAL_FAILURE(AssertPolicy(SCHED_FIFO, 89));
  }

  // confirm profile A is reverted
  EXPECT_NO_FATAL_FAILURE(AssertDefault());
}

}  // namespace ace_rt_profiles

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  // print logs to terminal for easier debugging
  FLAGS_logtostderr = true;
  google::InitGoogleLogging("ace_rt_profiles");

  return RUN_ALL_TESTS();
}
