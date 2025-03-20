cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-SIFT)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(sift-tracking sift-tracking.cpp)

# Link OpenCV libraries
target_link_libraries(sift-tracking ${OpenCV_LIBS})