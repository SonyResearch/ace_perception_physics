cmake_minimum_required(VERSION 3.5)
project(vision_common CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(ace_yaml REQUIRED)
find_package(base_parameters 1.0.0 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV 4.8 REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/vision_common.cpp src/ball_detection_parameters.cpp
                src/estimators/SpinEstimator.cpp src/estimators/VelocityEstimator.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_loggers base_parameters ace_yaml)
target_link_libraries(${PROJECT_NAME} glog gflags)

ament_target_dependencies(${PROJECT_NAME} OpenCV)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(base_parameters)

# pybind interface
ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(python_module SHARED src/python_module.cpp)

target_link_libraries(python_module PRIVATE ${PROJECT_NAME})

# install package
install(DIRECTORY include/ DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

# install cpython modules as sub-modules of the project
install(TARGETS python_module DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_test test/vision_common_test.cpp)

  add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})

  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})

  ament_target_dependencies(${PROJECT_NAME}_test)
endif()
ament_package()
