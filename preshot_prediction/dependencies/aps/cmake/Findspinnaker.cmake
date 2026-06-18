# FindSpinnaker.cmake — Locate FLIR Spinnaker SDK via pkg-config
# Provides imported target Spinnaker::Spinnaker

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(_SPINNAKER QUIET spinnaker)
endif()

find_path(Spinnaker_INCLUDE_DIR
  NAMES Spinnaker.h
  HINTS ${_SPINNAKER_INCLUDE_DIRS} /opt/spinnaker/include
)

find_library(Spinnaker_LIBRARY
  NAMES Spinnaker
  HINTS ${_SPINNAKER_LIBRARY_DIRS} /opt/spinnaker/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(spinnaker
  REQUIRED_VARS Spinnaker_LIBRARY Spinnaker_INCLUDE_DIR
)

if(spinnaker_FOUND AND NOT TARGET Spinnaker::Spinnaker)
  add_library(Spinnaker::Spinnaker SHARED IMPORTED)
  set_target_properties(Spinnaker::Spinnaker PROPERTIES
    IMPORTED_LOCATION "${Spinnaker_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Spinnaker_INCLUDE_DIR}"
  )
endif()

# For ament_target_dependencies compatibility
set(spinnaker_INCLUDE_DIRS ${Spinnaker_INCLUDE_DIR})
set(spinnaker_LIBRARIES ${Spinnaker_LIBRARY})
