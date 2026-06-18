cmake_minimum_required(VERSION 3.5)
project(triangulation CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(calibration 1.0.0 REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(ace_yaml 0.0.1 REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/triangulation.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_loggers)
target_link_libraries(${PROJECT_NAME} glog gflags)

ament_target_dependencies(${PROJECT_NAME} ace_yaml calibration OpenCV)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

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

# install cython modules as sub-modules of the project
install(TARGETS python_module DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# datalogger_pybind
pybind11_add_module(datalogger_pybind src/datalogger_pybind.cpp)
target_link_libraries(datalogger_pybind PRIVATE ${PROJECT_NAME} pybind11::module)
install(TARGETS datalogger_pybind DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  configure_file(test/root_dir_config.hpp test/root_dir.hpp)
  include_directories(${CMAKE_BINARY_DIR})
  ament_add_gtest(${PROJECT_NAME}_test test/triangulation_test.cpp)
  add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
  ament_target_dependencies(${PROJECT_NAME}_test)

  ament_add_gtest(test_${PROJECT_NAME}_datalogger test/test_datalogger.cpp)
  add_dependencies(test_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  target_link_libraries(test_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  ament_target_dependencies(test_${PROJECT_NAME}_datalogger)

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()
ament_package()
