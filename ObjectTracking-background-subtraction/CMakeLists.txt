cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-background-subtraction)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(background-subtraction-tracking background-subtraction-tracking.cpp)

# Link OpenCV libraries
target_link_libraries(background-subtraction-tracking ${OpenCV_LIBS})