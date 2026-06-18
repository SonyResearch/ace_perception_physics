cmake_minimum_required(VERSION 3.5)
project(dot_projector CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV 4.6 REQUIRED)
find_package(vision_common 1.0.0 REQUIRED)
find_package(calibration 1.0.0 REQUIRED)
find_package(ace_loggers REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/dot_projector.cpp)

ament_target_dependencies(${PROJECT_NAME} OpenCV vision_common calibration)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(vision_common calibration)

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
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_test test/dot_projector_test.cpp)
  add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME})
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
  ament_target_dependencies(${PROJECT_NAME}_test)

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()
ament_package()
