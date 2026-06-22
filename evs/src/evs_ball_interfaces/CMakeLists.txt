cmake_minimum_required(VERSION 3.5)
project(evs_ball_interfaces)

# find dependencies
#find_package(ace_cmake_utils REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# convert interfaces to language-specific code
rosidl_generate_interfaces(
  ${PROJECT_NAME}
  # messages
  "msg/Covariance.msg"
  "msg/ImagePoint.msg"
  "msg/PerceptionHeader.msg"
  "msg/PoseWithCovariance.msg"
  "msg/PosesWithCovariance.msg"
  "msg/VelocitiesImage.msg"
  "msg/VelocityImage.msg"
  DEPENDENCIES
  builtin_interfaces
  geometry_msgs
  sensor_msgs
  std_msgs)

ament_export_dependencies(rosidl_default_runtime)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E "copy_if_different" "${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}/__init__.py"
          "${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py/${PROJECT_NAME}/__init__.py")

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
