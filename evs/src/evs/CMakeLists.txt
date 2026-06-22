cmake_minimum_required(VERSION 3.5)
project(evs CXX)

# Enable pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)

# find dependencies
find_package(ace_cmake_utils REQUIRED)
ace_find_package(ament_cmake REQUIRED)
ace_find_package(ament_cmake_python REQUIRED)
ace_find_package(ace_interfaces 0.0.1 REQUIRED)
ace_find_package(base_parameters 1.0.0 REQUIRED)
ace_find_package(Eigen3 REQUIRED)
ace_find_package(ace_loggers REQUIRED)
ace_find_package(MetavisionSDK COMPONENTS core driver REQUIRED)
ace_find_package(MetavisionHAL REQUIRED)
ace_find_package(OpenCV 4.5 REQUIRED)
ace_find_package(rclcpp REQUIRED)
ace_find_package(rclpy REQUIRED)
ace_find_package(ace_yaml 0.0.1 REQUIRED)

# find python before pybind11
ace_find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
ace_find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
ace_add_library(${PROJECT_NAME} SHARED src/camera_parameters.cpp src/multi_camera_parameters.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_yaml base_parameters)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

# set executable
add_executable(metavision_recorder_standalone src/metavision_recorder.cpp src/metavision_recorder_standalone.cpp)
ament_target_dependencies(metavision_recorder_standalone ace_loggers)
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
  ace_find_package(ament_lint_auto REQUIRED)
  ace_find_package(ament_cmake_pytest REQUIRED)

  ament_lint_auto_find_test_dependencies()

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()

ament_package()
