cmake_minimum_required(VERSION 2.6)

project(arcface)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/ahjz/TensorRT-8.5.2.2/include)
link_directories(/home/ahjz/TensorRT-8.5.2.2/lib)

#jsoncpp
include_directories(/usr/include/jsoncpp)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
# cuda_add_library(myplugins SHARED yololayer.cu)
# target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#cuda_add_executable(yolov5 calibrator.cpp yolov5.cpp preprocess.cu)
cuda_add_executable(arcface_recog arcface_v2.cpp get_post.cpp)

target_link_libraries(arcface_recog nvinfer)
target_link_libraries(arcface_recog cudart)
# target_link_libraries(arcface myplugins)
target_link_libraries(arcface_recog ${OpenCV_LIBS} libjsoncpp.so)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


