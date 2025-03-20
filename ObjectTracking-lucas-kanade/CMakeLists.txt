
cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-lucas-kanade)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(lucas-kanade-tracking lucas-kanade-tracking.cpp)

# Link OpenCV libraries
target_link_libraries(lucas-kanade-tracking ${OpenCV_LIBS})