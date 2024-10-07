cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-sift)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(sift-tracking-exe sift-tracking.cpp)

# Link OpenCV libraries
target_link_libraries(sift-tracking-exe ${OpenCV_LIBS})
