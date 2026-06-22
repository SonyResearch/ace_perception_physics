cmake_minimum_required(VERSION 3.5)
project(data_generation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install top-level scripts
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

install(DIRECTORY tools DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

ament_package()
