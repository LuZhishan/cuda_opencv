cmake_minimum_required(VERSION 3.5)
project(cuda_01)

find_package(OpenCV 4 REQUIRED)
find_package(CUDA)

include_directories(${OpenCV_INCLUDE_DIRS})

# cuda_add_executable(01_add src/01_add.cu)
# target_link_libraries(01_add ${OpenCV_LIBS})
# cuda_add_executable(02_add_array src/02_add_array.cu)
# target_link_libraries(02_add_array ${OpenCV_LIBS})
cuda_add_executable(03_square src/03_square.cu)
target_link_libraries(03_square ${OpenCV_LIBS})