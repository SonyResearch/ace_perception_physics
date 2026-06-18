cmake_minimum_required(VERSION 3.16)
project(ace_monitor CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(ace_interfaces REQUIRED)
find_package(calibration REQUIRED)
find_package(ace_yaml REQUIRED)
find_package(ace_loggers REQUIRED)
find_package(vision_common REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Freetype REQUIRED)
find_package(assimp REQUIRED)
find_package(yaml-cpp REQUIRED)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# ---------------------------------------------------------------------------
# Vendored UI libraries (Dear ImGui + ImPlot) via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.4
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG v0.16
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui implot)

add_library(imgui_impl STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  ${implot_SOURCE_DIR}/implot_demo.cpp
)
target_include_directories(imgui_impl SYSTEM PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${implot_SOURCE_DIR}
)
target_link_libraries(imgui_impl PUBLIC glfw OpenGL::GL)

# ---------------------------------------------------------------------------
# Includes
# ---------------------------------------------------------------------------
include_directories(include)
include_directories(SYSTEM external/shims/include)
include_directories(SYSTEM /usr/include/stb)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(SYSTEM ${FREETYPE_INCLUDE_DIRS})
include_directories(SYSTEM ${assimp_INCLUDE_DIRS})
# Force the vendored Dear ImGui / ImPlot headers to win over any
# system-installed copy (e.g. /usr/local/include/imgui).
include_directories(BEFORE
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${implot_SOURCE_DIR}
)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED
  src/pages/IMonitorPage.cpp
  src/pages/CameraMonitorPage.cpp
  src/pages/BallDetectorPage.cpp
  src/pages/BallTriangulationPage.cpp
  src/pages/BallPoseEstimationPage.cpp
  src/pages/PerceptionVisualizerPage.cpp
  src/pages/PlayerPoseEstimationPage.cpp
  src/pages/RacketPoseEstimationPage.cpp
  src/pages/EstimatorTopicsPage.cpp
  src/pages/BallEstimatorPage.cpp
  src/pages/PerceptionDelayPage.cpp
  src/pages/BallSpinPage.cpp
  src/pages/AlarmsPage.cpp
  src/AceMonitorRosNode.cpp
  src/scene/Scene3D.cpp
  src/scene/Node3D.cpp
  src/scene/Nodes.cpp
  src/scene/TextRenderer.cpp
  src/scene/rendering/GLTexture.cpp
  src/scene/rendering/GLShader.cpp
  src/scene/rendering/ShaderUniforms.cpp
  src/scene/rendering/Material.cpp
  src/scene/rendering/Mesh3D.cpp
  src/scene/rendering/Model3D.cpp
  src/scene/rendering/SceneResources.cpp
  src/utils/TextureHelpers.cpp
  src/utils/common_values.cpp
  src/utils/Geometry.cpp
  src/utils/MeshGenerator.cpp
  src/AceMonitor.cpp
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  imgui_impl
  glfw
  GLEW::GLEW
  OpenGL::GL
  Eigen3::Eigen
  ${OpenCV_LIBS}
  ${FREETYPE_LIBRARIES}
  assimp
  yaml-cpp
)

ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp
  ament_index_cpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  builtin_interfaces
  ace_interfaces
  ace_yaml
  ace_loggers
  calibration
  ace_rt_profiles
  vision_common
)

# ---------------------------------------------------------------------------
# Standalone executable
# ---------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_standalone src/ace_monitor_standalone.cpp)
target_link_libraries(${PROJECT_NAME}_standalone PRIVATE ${PROJECT_NAME})

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME}_standalone DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch data DESTINATION share/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

ament_export_dependencies(rclcpp ament_index_cpp ace_interfaces)
ament_package()
