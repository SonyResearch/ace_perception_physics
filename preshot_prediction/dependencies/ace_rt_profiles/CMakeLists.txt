cmake_minimum_required(VERSION 3.5)
project(ace_rt_profiles CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ace_yaml 0.1.21 REQUIRED)

# can't depend on ace_loggers due to circular dependency
find_package(glog REQUIRED)

# find pybind
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

add_library(${PROJECT_NAME} SHARED src/profile_manager.cpp src/profile_guard.cpp src/utils.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
ament_target_dependencies(${PROJECT_NAME} PUBLIC ament_index_cpp ace_yaml)
target_link_libraries(${PROJECT_NAME} PRIVATE glog::glog)
target_link_options(${PROJECT_NAME} PRIVATE -Wl,--no-undefined)

pybind11_add_module(${PROJECT_NAME}_bindings src/bindings.cpp)
target_link_libraries(${PROJECT_NAME}_bindings PRIVATE ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_bindings PROPERTIES OUTPUT_NAME "bindings")

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}_bindings DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})
install(PROGRAMS scripts/cat_ace_rt_profiles DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(pytest_${PROJECT_NAME} test/test.py)
  ament_add_gmock(test_${PROJECT_NAME} test/test.cpp)
  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME} glog::glog)

  install(DIRECTORY test/config/ DESTINATION share/${PROJECT_NAME}/config/test)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(ament_index_cpp glog ace_yaml)

ament_package()
