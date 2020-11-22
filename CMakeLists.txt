cmake_minimum_required(VERSION 2.8)
project(feature_extraction)

# ============================================================
# release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# ============================================================
# find packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(gflags REQUIRED)

find_path(CUDA_INCLUDE_DIR cuda_runtime.h
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_path(CUDA_LIBRARY_DIR libcudart.so
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

message(STATUS "  include path: ${CUDA_INCLUDE_DIR}")
message(STATUS "  library path: ${CUDA_LIBRARY_DIR}")

# ============================================================
# find tensorrt
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

if(NOT (CUDA_INCLUDE_DIR AND CUDA_LIBRARY_DIR AND
    TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER AND
    TENSORRT_LIBRARY_INFER_PLUGIN AND TENSORRT_LIBRARY_ONNXPARSER))
  message(FATAL_ERROR "CUDA or TensorRT NOT FOUND")
endif()

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER}
                     ${TENSORRT_LIBRARY_INFER_PLUGIN}
                     ${TENSORRT_LIBRARY_ONNXPARSER})

# ============================================================
# include 
include_directories(${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} 
  ${YAML_CPP_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIR} 
  ${PROJECT_SOURCE_DIR}/common
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/models)

# ============================================================
# executable 

add_executable(
  inference 
  src/main.cpp 
  src/base_model.cpp 
  src/preprocess.cpp 
  src/stereo_camera.cpp
  src/models/stereo_depth.cpp
  src/models/instance_seg.cpp
  src/models/feature_extraction.cpp
  src/models/mono_depth.cpp
  src/models/yolov5.cpp
  common/logger.cpp 
)

target_link_libraries(inference ${OpenCV_LIBS}
  ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} yaml-cpp gflags )


add_executable(
  test_preprocess 
  test/test_preprocess.cpp 
  src/preprocess.cpp 
  src/stereo_camera.cpp
)

target_link_libraries(test_preprocess ${OpenCV_LIBS}
  ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} yaml-cpp gflags )