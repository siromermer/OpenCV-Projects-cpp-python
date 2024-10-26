cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-kalman)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(kalman-tracking-exe kalman-tracking-predicting.cpp)

# Link OpenCV libraries
target_link_libraries(kalman-tracking-exe ${OpenCV_LIBS})
