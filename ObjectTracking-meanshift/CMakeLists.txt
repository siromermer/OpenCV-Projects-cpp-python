cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-meanshift)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(meanshift-tracking meanshift-tracking.cpp)

# Link OpenCV libraries
target_link_libraries(meanshift-tracking ${OpenCV_LIBS})