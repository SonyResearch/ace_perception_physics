cmake_minimum_required(VERSION 3.5)
project(ball_detection_aps)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_link_options(-Wl,--no-undefined)

# find dependencies
find_package(ace_interfaces 0.0.1 REQUIRED)
find_package(calibration 1.0.0 REQUIRED)
find_package(cuda_common REQUIRED)
find_package(vision_common 1.0.4 REQUIRED)
find_package(ball_detector 1.0.0 REQUIRED)
find_package(dot_detector REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ace_rt_profiles REQUIRED)
find_package(ace_yaml 0.0.1 REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/ball_detection_aps_parameters.cpp src/ball_detection_aps.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_loggers)
target_link_libraries(${PROJECT_NAME} glog gflags)
target_link_libraries(${PROJECT_NAME})

ament_target_dependencies(
  ${PROJECT_NAME}
  ace_yaml
  ace_interfaces
  calibration
  vision_common
  cuda_common
  ball_detector
  dot_detector
  ace_rt_profiles
  OpenCV
  rclcpp)

# set executable
add_executable(${PROJECT_NAME}_standalone src/ball_detection_aps_standalone.cpp)
ament_target_dependencies(${PROJECT_NAME}_standalone ace_loggers)
target_link_libraries(${PROJECT_NAME}_standalone glog gflags)

add_dependencies(${PROJECT_NAME}_standalone ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}_standalone ${PROJECT_NAME})

ament_target_dependencies(${PROJECT_NAME}_standalone rclcpp)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

# pybind interface
ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(python_module SHARED src/python_module.cpp)

target_link_libraries(python_module PRIVATE ${PROJECT_NAME})

ament_target_dependencies(python_module PUBLIC OpenCV)

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

install(TARGETS ${PROJECT_NAME}_standalone DESTINATION lib/${PROJECT_NAME})

# install cython modules as sub-modules of the project
install(TARGETS python_module DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

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
