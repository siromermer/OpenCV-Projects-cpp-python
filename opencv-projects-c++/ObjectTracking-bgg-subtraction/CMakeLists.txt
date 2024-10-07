cmake_minimum_required(VERSION 3.10)
project(ObjectTracking-bgg-subtraction)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(bgg-subtraction-exe bgg-subtraction.cpp)

# Link OpenCV libraries
target_link_libraries(bgg-subtraction-exe ${OpenCV_LIBS})
