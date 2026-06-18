cmake_minimum_required(VERSION 3.5)
project(ace_loggers)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ace_rt_profiles REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(Boost REQUIRED)

find_package(nlohmann_json REQUIRED)
if(nlohmann_json_FOUND)
  include_directories(${nlohmann_json_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "nlohmann_json NOT Found!")
endif()

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/logger.cpp src/SignalHandler.cpp)
ament_target_dependencies(${PROJECT_NAME} ace_rt_profiles)
ament_export_targets("export_${PROJECT_NAME}" HAS_LIBRARY_TARGET)
ament_export_dependencies(ace_rt_profiles glog gflags Boost)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT "export_${PROJECT_NAME}"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  INCLUDES
  DESTINATION include)

install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME})
# Install vendored ace_helpers stub header (no upstream ace_helpers package available)
install(DIRECTORY include/ace_helpers/ DESTINATION include/ace_helpers)
ament_export_include_directories(include)

# logger_pybind
pybind11_add_module(logger_pybind src/logger_pybind.cpp)
target_link_libraries(logger_pybind PRIVATE ${PROJECT_NAME} pybind11::module glog::glog)
install(TARGETS logger_pybind DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# datalogger_pybind
pybind11_add_module(datalogger_pybind src/datalogger_pybind.cpp)
target_link_libraries(datalogger_pybind PRIVATE ${PROJECT_NAME} pybind11::module)
install(TARGETS datalogger_pybind DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# datalogger_demo_pybind
pybind11_add_module(datalogger_demo_pybind src/datalogger_demo_pybind.cpp)
target_link_libraries(datalogger_demo_pybind PRIVATE ${PROJECT_NAME} pybind11::module)
install(TARGETS datalogger_demo_pybind DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")

# benchmark
if(BUILD_BENCHMARK)
  find_package(ament_cmake_google_benchmark REQUIRED)

  ament_add_google_benchmark(benchmark_${PROJECT_NAME}_datalogger test/benchmark_datalogger.cpp)
  target_compile_definitions(benchmark_${PROJECT_NAME}_datalogger PUBLIC DISABLE_THREADING=1)
  add_dependencies(benchmark_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  target_link_libraries(benchmark_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  ament_target_dependencies(benchmark_${PROJECT_NAME}_datalogger)

  ament_add_google_benchmark(benchmark_${PROJECT_NAME}_datalogger_threaded test/benchmark_datalogger.cpp)
  add_dependencies(benchmark_${PROJECT_NAME}_datalogger_threaded ${PROJECT_NAME})
  target_link_libraries(benchmark_${PROJECT_NAME}_datalogger_threaded ${PROJECT_NAME})
  ament_target_dependencies(benchmark_${PROJECT_NAME}_datalogger_threaded)
endif()

# testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_${PROJECT_NAME}_datalogger test/test_datalogger.cpp)
  add_dependencies(test_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  target_link_libraries(test_${PROJECT_NAME}_datalogger ${PROJECT_NAME})
  ament_target_dependencies(test_${PROJECT_NAME}_datalogger)

  ament_add_gtest(test_${PROJECT_NAME}_datalogger_demo test/test_datalogger_demo.cpp)
  add_dependencies(test_${PROJECT_NAME}_datalogger_demo ${PROJECT_NAME})
  target_link_libraries(test_${PROJECT_NAME}_datalogger_demo ${PROJECT_NAME})
  ament_target_dependencies(test_${PROJECT_NAME}_datalogger_demo)

  file(GLOB_RECURSE PY_TEST_FILES "${PROJECT_SOURCE_DIR}/test/test_*.py")
  foreach(TEST ${PY_TEST_FILES})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    ament_add_pytest_test(${PROJECT_NAME}_${TEST_NAME} ${TEST} APPEND_LIBRARY_DIRS
                          ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
  endforeach()
endif()
ament_package()
