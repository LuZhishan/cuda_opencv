cmake_minimum_required(VERSION 3.5)
project(cuda_02)

find_package(OpenCV 4 REQUIRED)
find_package(CUDA)

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(01_add_img src/01_add_img.cpp)
target_link_libraries(01_add_img ${OpenCV_LIBS})