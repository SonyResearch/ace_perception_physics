cmake_minimum_required(VERSION 3.5)
project(evs_ball_cpp)

add_link_options(-Wl,--no-undefined)

# find dependencies
find_package(evs_ball_interfaces 0.0.1 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(base_parameters 1.0.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(glog REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# find CUDA libraries
find_package(CUDAToolkit REQUIRED)

# set up include and link directories
include_directories(include)
include_directories(SYSTEM external)

link_directories(${CMAKE_CUDA_INCLUDE_DIRS})

# set library
add_library(
  ${PROJECT_NAME}
  SHARED
  src/velocity_evs_parameters.cpp
  src/frame_generation.cpp
  src/velocity_prediction_evs.cpp
  src/imageCHW.cpp
  src/trt_engine.cpp)

ament_target_dependencies(${PROJECT_NAME} glog)
target_link_libraries(${PROJECT_NAME} glog gflags)
target_link_libraries(
  ${PROJECT_NAME}
  CUDA::cudart_static
  CUDA::nvml
  MetavisionSDK::core
  MetavisionSDK::driver
  ${OpenCV_LIBS}
  nvinfer
  nvinfer_plugin)

ament_target_dependencies(
  ${PROJECT_NAME}
  evs_ball_interfaces
  base_parameters
  OpenCV
  rclcpp)

# set executable
add_executable(${PROJECT_NAME}_standalone src/velocity_prediction_evs_standalone.cpp)

add_dependencies(${PROJECT_NAME}_standalone ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}_standalone ${PROJECT_NAME})

ament_target_dependencies(${PROJECT_NAME}_standalone rclcpp)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(CUDAToolkit)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# install package
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY external/ DESTINATION external)

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

ament_package()
