cmake_minimum_required(VERSION 3.5)
project(base_parameters CXX)

# find dependencies
find_package(ace_cmake_utils REQUIRED)
ace_find_package(ament_cmake REQUIRED)
ace_find_package(ace_loggers REQUIRED)
ace_find_package(ace_yaml 0.0.1 REQUIRED)

# find python before pybind11
ace_find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
ace_find_package(pybind11 CONFIG REQUIRED)

# set include directories
include_directories(include)

# set library
ace_add_library(${PROJECT_NAME} SHARED src/base_parameters.cpp)

ament_target_dependencies(${PROJECT_NAME} ace_loggers)
target_link_libraries(${PROJECT_NAME} glog gflags)

ament_target_dependencies(${PROJECT_NAME} ace_yaml)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

ament_export_dependencies(ace_yaml)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

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

if(BUILD_TESTING)
  ace_find_package(ament_cmake_pytest REQUIRED)

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()

ament_package()
