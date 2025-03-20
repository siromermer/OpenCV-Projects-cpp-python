cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-orb)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(orb-tracking orb-tracking.cpp)

# Link OpenCV libraries
target_link_libraries(orb-tracking ${OpenCV_LIBS})