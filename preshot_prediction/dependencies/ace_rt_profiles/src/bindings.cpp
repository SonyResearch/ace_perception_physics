// Confidential, Copyright 2025, Sony AI, All rights reserved

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ace_rt_profiles/profile_guard.hpp"
#include "ace_rt_profiles/profile_manager.hpp"
#include "ace_rt_profiles/rt_sub_profile.hpp"
#include "ace_rt_profiles/utils.hpp"

namespace py = pybind11;

namespace ace_rt_profiles {

PYBIND11_MODULE(bindings, m) {
  py::class_<ProfileManager, std::unique_ptr<ProfileManager, py::nodelete>>(m, "ProfileManager")
    .def_static("get_instance", &ProfileManager::GetInstance, py::return_value_policy::reference)
    .def("apply", py::overload_cast<const std::string&>(&ProfileManager::Apply, py::const_))
    .def("apply", py::overload_cast<int, const std::string&>(&ProfileManager::Apply, py::const_))
    .def("apply", py::overload_cast<const RTSubProfile&>(&ProfileManager::Apply, py::const_))
    .def("apply", py::overload_cast<int, const RTSubProfile&>(&ProfileManager::Apply, py::const_))
    .def("get_affinity", &ProfileManager::GetAffinity);

  py::class_<ProfileGuard>(m, "ProfileGuard")
    .def(py::init())
    .def("__enter__", [](const ProfileGuard& /* self */) {})
    .def("__exit__", [](const ProfileGuard& /* self */, py::args /* *exc */) {});

  py::class_<RTSubProfile>(m, "RTSubProfile")
    .def(py::init<const std::string&, size_t>(), py::arg("rt_profile") = "", py::arg("affinity_index") = 0)
    .def_readwrite("rt_profile", &RTSubProfile::rt_profile)
    .def_readwrite("affinity_index", &RTSubProfile::affinity_index);

  m.def("get_config_file", &utils::GetConfigFile);
}

}  // namespace ace_rt_profiles
