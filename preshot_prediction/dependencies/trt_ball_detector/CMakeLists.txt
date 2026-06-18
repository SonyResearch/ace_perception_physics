cmake_minimum_required(VERSION 3.5)
project("trt_ball_detector")

set(CMAKE_CXX_STANDARD 20)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(cuda_common REQUIRED)
find_package(vision_common REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED
            src/ball_detector.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_loggers ament_index_cpp cuda_common vision_common)
target_link_libraries(${PROJECT_NAME} glog gflags)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(ament_index_cpp)

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
install(TARGETS python_module
        DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_${PROJECT_NAME}
                  test/test_trt_ball_detector.cpp)
  configure_file(test/root_dir_config.hpp test/root_dir.hpp)
  target_include_directories(test_${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR})
  add_dependencies(test_${PROJECT_NAME} ${PROJECT_NAME})
  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
  ament_target_dependencies(test_${PROJECT_NAME} ace_loggers ament_index_cpp cuda_common vision_common)

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(
      ${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
      ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()
ament_package()
