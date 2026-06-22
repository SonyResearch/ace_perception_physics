cmake_minimum_required(VERSION 3.5)
project(evs CXX)

# Enable pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ace_interfaces 0.0.1 REQUIRED)
find_package(base_parameters 1.0.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver REQUIRED)
find_package(MetavisionHAL REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ace_yaml 0.0.1 REQUIRED)
find_package(Threads REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/camera_parameters.cpp src/multi_camera_parameters.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_yaml base_parameters)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

# set executable
add_executable(metavision_recorder_standalone src/metavision_recorder.cpp src/metavision_recorder_standalone.cpp)
target_link_libraries(metavision_recorder_standalone glog gflags)
add_dependencies(metavision_recorder_standalone ${PROJECT_NAME})

target_link_libraries(metavision_recorder_standalone ${PROJECT_NAME} Metavision::HAL_discovery Threads::Threads)
ament_target_dependencies(metavision_recorder_standalone ace_yaml rclcpp)

# pybind interface
ament_python_install_package(${PROJECT_NAME})

# install package
install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY launch parameters DESTINATION share/${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(TARGETS metavision_recorder_standalone DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_lint_auto_find_test_dependencies()

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()

ament_package()
