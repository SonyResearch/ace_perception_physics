cmake_minimum_required(VERSION 3.5)
project(data_generation)

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Compute site-packages path relative to install prefix
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import sys; print('lib/python{}.{}/site-packages'.format(*sys.version_info[:2]))"
  OUTPUT_VARIABLE PYTHON_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Install the data_generation directory as an importable Python package so
# that "from data_generation.tools.xxx import ..." works after colcon install.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION "${PYTHON_INSTALL_DIR}/data_generation"
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".git" EXCLUDE
)

# Install top-level scripts as executables
install(PROGRAMS
  label_pro_player_data.py
  label_pro_player_data_events.py
  process_raw_files.py
  convert_h5_to_video.py
  convert_h5_to_video_zoomed.py
  trim_h5_sequences.py
  visualise_representations_h5.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
