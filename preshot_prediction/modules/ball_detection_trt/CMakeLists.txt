cmake_minimum_required(VERSION 3.5)
project("ball_detection_trt")

set(CMAKE_CXX_STANDARD 20)
add_link_options(-Wl,--no-undefined)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(ace_interfaces 0.0.1 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ace_yaml 0.0.1 REQUIRED)
find_package(base_parameters 1.0.0 REQUIRED)
find_package(cuda_common REQUIRED)
find_package(vision_common REQUIRED)
find_package(calibration REQUIRED)
find_package(trt_ball_detector REQUIRED)
find_package(ace_rt_profiles REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED
            src/ball_detection_trt.cpp
            src/ball_detection_parameters.cpp
            )

ament_target_dependencies(${PROJECT_NAME} ace_loggers ace_yaml ace_interfaces ace_rt_profiles
                          base_parameters rclcpp trt_ball_detector calibration)
target_link_libraries(${PROJECT_NAME} glog gflags)

add_executable(${PROJECT_NAME}_standalone
               src/ball_detection_trt_standalone.cpp)
add_dependencies(${PROJECT_NAME}_standalone ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_standalone ${PROJECT_NAME})

ament_target_dependencies(${PROJECT_NAME}_standalone rclcpp)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(ament_index_cpp)
# install package
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch parameters DESTINATION share/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}_standalone DESTINATION lib/${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

ament_package()
