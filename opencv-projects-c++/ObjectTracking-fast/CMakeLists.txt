cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-fast)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(fast-tracking-exe fast_tracking.cpp)

# Link OpenCV libraries
target_link_libraries(fast-tracking-exe  ${OpenCV_LIBS})
