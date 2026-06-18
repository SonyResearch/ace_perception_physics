cmake_minimum_required(VERSION 3.18)

# Force CUDA host compiler to a version compatible with this toolkit (CUDA 11.x
# only supports up to gcc-11). Allow override via -DCMAKE_CUDA_HOST_COMPILER.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  if(EXISTS /usr/bin/g++-10)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)
  elseif(EXISTS /usr/bin/g++-11)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)
  endif()
endif()

project(cuda_common LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_link_options(-Wl,--no-undefined)

# NVCC flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(vision_common REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV 4.8 REQUIRED)

# find python before pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# find CUDA libraries
find_package(CUDAToolkit REQUIRED)

# set up include and link directories
include_directories(include)

# set library
add_library(${PROJECT_NAME} SHARED src/cuda_common.cpp src/TRTEngine.cpp src/bayer_resize.cu)

target_link_libraries(${PROJECT_NAME} glog gflags)
target_link_libraries(${PROJECT_NAME} CUDA::cudart_static CUDA::nvml nvonnxparser nvinfer nvinfer_plugin)

ament_target_dependencies(${PROJECT_NAME} ace_loggers vision_common OpenCV)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(base_parameters CUDAToolkit)

# pybind interface
ament_python_install_package(${PROJECT_NAME})

pybind11_add_module(python_module SHARED src/python_module.cpp)

target_link_libraries(python_module PRIVATE ${PROJECT_NAME})

# install package
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY test DESTINATION share/${PROJECT_NAME})

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
