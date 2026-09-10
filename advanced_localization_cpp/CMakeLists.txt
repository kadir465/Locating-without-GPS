cmake_minimum_required(VERSION 3.16)
project(advanced_localization_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio highgui calib3d dnn)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(GSTREAMER QUIET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
endif()

option(ADVANCED_LOCALIZATION_BUILD_LOCALIZATION_NODE "Build the ROS 2 PX4 localization_node target" ON)

set(ADVANCED_LOCALIZATION_HAS_PX4_VISUAL_ODOMETRY FALSE)
if(ADVANCED_LOCALIZATION_BUILD_LOCALIZATION_NODE)
  find_path(PX4_VEHICLE_VISUAL_ODOMETRY_INCLUDE_DIR
    NAMES px4_msgs/msg/vehicle_visual_odometry.hpp
    PATHS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include include/px4_msgs
  )
  if(PX4_VEHICLE_VISUAL_ODOMETRY_INCLUDE_DIR)
    set(ADVANCED_LOCALIZATION_HAS_PX4_VISUAL_ODOMETRY TRUE)
  else()
    message(WARNING
      "px4_msgs/msg/vehicle_visual_odometry.hpp was not found. "
      "Skipping localization_node so video_testing_runner can still build. "
      "Use a px4_msgs branch matching the deployed PX4 firmware before building the PX4 ROS node."
    )
  endif()
endif()

include_directories(include)

set(LIB_SOURCES
  src/config.cpp
  src/eskf.cpp
  src/failsafe.cpp
  src/map_store.cpp
  src/math_utils.cpp
  src/optical_flow_layer.cpp
  src/roi_selector.cpp
  src/video_testing.cpp
  src/visual_localizer.cpp
)

add_library(${PROJECT_NAME}_core ${LIB_SOURCES})
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS} yaml-cpp Threads::Threads)
if(GSTREAMER_FOUND)
  target_compile_definitions(${PROJECT_NAME}_core PUBLIC ADVANCED_LOCALIZATION_HAS_GSTREAMER=1)
  target_include_directories(${PROJECT_NAME}_core PUBLIC ${GSTREAMER_INCLUDE_DIRS})
  target_link_directories(${PROJECT_NAME}_core PUBLIC ${GSTREAMER_LIBRARY_DIRS})
  target_link_libraries(${PROJECT_NAME}_core PUBLIC ${GSTREAMER_LIBRARIES})
  target_compile_options(${PROJECT_NAME}_core PUBLIC ${GSTREAMER_CFLAGS_OTHER})
endif()

add_executable(video_testing_runner src/video_testing_runner.cpp)
target_link_libraries(video_testing_runner PRIVATE ${PROJECT_NAME}_core)

set(ADVANCED_LOCALIZATION_INSTALL_TARGETS ${PROJECT_NAME}_core video_testing_runner)

if(ADVANCED_LOCALIZATION_BUILD_LOCALIZATION_NODE AND ADVANCED_LOCALIZATION_HAS_PX4_VISUAL_ODOMETRY)
  add_executable(localization_node src/localization_node.cpp)
  target_link_libraries(localization_node ${PROJECT_NAME}_core)
  ament_target_dependencies(localization_node rclcpp sensor_msgs std_msgs px4_msgs)
  list(APPEND ADVANCED_LOCALIZATION_INSTALL_TARGETS localization_node)
endif()

install(TARGETS ${ADVANCED_LOCALIZATION_INSTALL_TARGETS}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README_VIDEO_TESTING.md DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core
    test/test_core.cpp
  )
  target_link_libraries(test_core ${PROJECT_NAME}_core)
endif()

ament_package()
