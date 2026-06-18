cmake_minimum_required(VERSION 3.5)
project(aps)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_link_options(-Wl,--no-undefined)

# Add local cmake module path for FindSpinnaker
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find dependencies
find_package(ace_interfaces 0.0.1 REQUIRED)
find_package(ace_rt_profiles REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(base_parameters 1.0.0 REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(vision_common 1.0.0 REQUIRED)
find_package(cuda_common REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(calibration REQUIRED)
find_package(ace_yaml 0.0.1 REQUIRED)
find_package(spinnaker REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# find cuda libraries
find_package(CUDAToolkit REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(
  ${PROJECT_NAME}
  SHARED
  src/blackfly_s_camera.cpp
  src/blackfly_s_image_event_handler.cpp
  src/blackfly_s_device_event_handler.cpp
  src/blackfly_s_interface_event_handler.cpp
  src/camera_parameters.cpp
  src/jpeg_encoder.cpp
  src/multi_camera_parameters.cpp
  src/multi_camera.cpp
  src/recorder_parameters.cpp
  src/recorder.cpp
  src/single_camera.cpp
  src/SyncedImagesCallback.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_rt_profiles ace_loggers)
target_link_libraries(${PROJECT_NAME} glog gflags)
target_link_libraries(${PROJECT_NAME} CUDA::cudart_static CUDA::culibos CUDA::nvml nvjpeg)

ament_target_dependencies(
  ${PROJECT_NAME}
  ace_interfaces
  base_parameters
  cuda_common
  vision_common
  OpenCV
  rclcpp
  std_msgs
  ace_yaml
  calibration)

# Link Spinnaker separately (not an ament package)
target_include_directories(${PROJECT_NAME} PUBLIC ${spinnaker_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${spinnaker_LIBRARIES})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(vision_common CUDAToolkit ace_rt_profiles)

# set executable
add_executable(multi_camera_standalone src/multi_camera_standalone.cpp)

add_dependencies(multi_camera_standalone ${PROJECT_NAME})

ament_target_dependencies(multi_camera_standalone ace_loggers)
target_link_libraries(multi_camera_standalone glog gflags)
target_link_libraries(multi_camera_standalone ${PROJECT_NAME})

ament_target_dependencies(multi_camera_standalone rclcpp)

add_executable(recorder_standalone src/recorder_standalone.cpp)

add_dependencies(recorder_standalone ${PROJECT_NAME})

ament_target_dependencies(recorder_standalone ace_loggers)
target_link_libraries(recorder_standalone glog gflags)
target_link_libraries(recorder_standalone ${PROJECT_NAME})

ament_target_dependencies(recorder_standalone rclcpp)

add_executable(single_camera_standalone src/single_camera_standalone.cpp)

add_dependencies(single_camera_standalone ${PROJECT_NAME})

ament_target_dependencies(single_camera_standalone ace_loggers)
target_link_libraries(single_camera_standalone glog gflags)
target_link_libraries(single_camera_standalone ${PROJECT_NAME})

ament_target_dependencies(single_camera_standalone rclcpp)

# pybind interface
ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(python_module SHARED src/python_module.cpp src/zero_copy_subscriber.cpp)

target_link_libraries(python_module PRIVATE ${PROJECT_NAME})

ament_target_dependencies(python_module PUBLIC base_parameters)

# install package
install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY tools launch parameters DESTINATION share/${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(TARGETS multi_camera_standalone recorder_standalone single_camera_standalone DESTINATION lib/${PROJECT_NAME})

# install cython modules as sub-modules of the project
install(TARGETS python_module DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# scripts meant to be executed without calling ros2 run
install(PROGRAMS tools/aps_multi_camera_cpu_setup tools/aps_play_publisher tools/aps_rec_publisher
                 tools/aps_show_camera_images DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()

ament_package()
