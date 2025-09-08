cmake_minimum_required(VERSION 3.10)
project(YOLO-inference-cpp) # your folder name here

# Find OpenCV
set(OpenCV_DIR C:/Libraries/opencv/build) # path to opencv
find_package(OpenCV REQUIRED)

add_executable(object_detection object_detection.cpp) # your file name

# Link OpenCV libraries
target_link_libraries(object_detection ${OpenCV_LIBS})