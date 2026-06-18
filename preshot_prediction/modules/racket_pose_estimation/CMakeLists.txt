cmake_minimum_required(VERSION 3.5)
project(racket_pose_estimation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_link_options(-Wl,--no-undefined)

# find dependencies
find_package(ace_interfaces 0.0.1 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(ace_yaml 0.0.1 REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(calibration REQUIRED)
find_package(vision_common REQUIRED)
find_package(cuda_common REQUIRED)
find_package(triangulation REQUIRED)
find_package(aps REQUIRED)
find_package(ace_rt_profiles REQUIRED)
find_package(OpenCV 4.9 REQUIRED)
find_package(GTest REQUIRED)
find_package(rclcpp REQUIRED)

# set up include and link directories
include_directories(include)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set library
add_library(
  ${PROJECT_NAME}
  SHARED
  src/RacketPoseFitter.cpp
  src/RacketParameters.cpp
  src/RacketPoseExtractor.cpp
  src/RacketInference.cpp
  # ROS
  src/ros/RacketPoseEstimator.cpp
  src/ros/RacketPoseROSNode.cpp)

target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin nvonnxparser)

ament_target_dependencies(
  ${PROJECT_NAME}
  ace_yaml
  aps
  ace_loggers
  ace_rt_profiles
  calibration
  ace_interfaces
  vision_common
  cuda_common
  triangulation
  base_parameters
  rclcpp
  OpenCV)

# pybind interface
ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(python_module SHARED src/python_module.cpp)

target_link_libraries(python_module PRIVATE ${PROJECT_NAME})

# set executable
add_executable(${PROJECT_NAME}_standalone src/racket_pose_estimation_standalone.cpp)

add_dependencies(${PROJECT_NAME}_standalone ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}_standalone ${PROJECT_NAME})
ament_target_dependencies(${PROJECT_NAME}_standalone)

# install package
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY parameters models launch scripts DESTINATION share/${PROJECT_NAME})
install(PROGRAMS bin/racket_pose_filter DESTINATION lib/${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(TARGETS ${PROJECT_NAME}_standalone DESTINATION lib/${PROJECT_NAME})

# install cython modules as sub-modules of the project
install(TARGETS python_module DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# datalogger_pybind
pybind11_add_module(datalogger_pybind src/datalogger_pybind.cpp)
target_link_libraries(datalogger_pybind PRIVATE ${PROJECT_NAME} pybind11::module)
install(TARGETS datalogger_pybind DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# TESTING
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  install(DIRECTORY tests DESTINATION share/${PROJECT_NAME})

  ament_add_gtest(test_${PROJECT_NAME}_datalogger tests/test_datalogger.cpp)
  add_dependencies(test_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  target_link_libraries(test_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  ament_target_dependencies(test_${PROJECT_NAME}_datalogger ament_index_cpp)

  find_package(ament_cmake_pytest REQUIRED)
  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/tests/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()

endif()

ament_package()
