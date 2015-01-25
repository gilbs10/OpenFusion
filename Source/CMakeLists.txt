cmake_minimum_required (VERSION 2.6)
project (OpenFusion)

find_package(GLEW REQUIRED)

FIND_PACKAGE(CUDA REQUIRED)

option (BUILD_FROMFILE "Build the camera from file library" ON) 
option (BUILD_SOFTKINETIC "Build the softkinetic camera library" OFF) 
option (BUILD_PRIMESENSE "Build the primesense camera library" OFF) 
option (BUILD_GIPCAM "Build the gipcam camera library" OFF)

set(SOFTKINETIC_SDK_LIB "DepthSense lib not found" CACHE FILEPATH "The softkinetic sdk lib")
set(OPENNI_LIB "OpenNI lib not found" CACHE FILEPATH "The OpenNI lib")
set(SOFTKINETIC_SDK_INCLUDE "DepthSense include path not found" CACHE PATH "The softkinetic sdk include path")
set(OPENNI_INCLUDE "OpenNI lib include path found" CACHE PATH "The OpenNI include path")

configure_file( ${CMAKE_SOURCE_DIR}/include/config.h.in ${CMAKE_SOURCE_DIR}/include/config.h )

add_subdirectory(src)

INCLUDE(FindCUDA)

FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h")
#CUDA_ADD_EXECUTABLE(lbmslv ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O3 -ffast-math")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)



FIND_PATH(
	CIMG_INCLUDE_DIR CImg.h
	"${CMAKE_CURRENT_SOURCE_DIR}/CImg"
         )

SET(CIMG_FOUND FALSE)
IF(CIMG_INCLUDE_DIR)
  SET(CIMG_FOUND TRUE)
  INCLUDE_DIRECTORIES(${CIMG_INCLUDE_DIR})
ELSE(CIMG_INCLUDE_DIR)
  MESSAGE(FATAL ERROR "CImg.h not found - Please install CImg")
ENDIF(CIMG_INCLUDE_DIR)


include_directories (
${GLEW_INCLUDE_DIR}
"${PROJECT_SOURCE_DIR}/include"
"${PROJECT_SOURCE_DIR}/include/TinyThread"
"${PROJECT_SOURCE_DIR}/include/Eigen"
"${PROJECT_SOURCE_DIR}/include/Utils"
${CUDA_TOOLKIT_INCLUDE}
${OPENNI_INCLUDE}
${SOFTKINETIC_SDK_INCLUDE})


add_executable(OpenFusion src/MainApp.cpp)


target_link_libraries (OpenFusion 
OpenFusionSrc
OpenFusionSrcCU
OpenFusionCam
OpenFusionCamCU
OpenFusionUtils
${GLEW_LIBRARY}
${CUDA_CUDART_LIBRARY}
pthread
X11
GL
GLU
glut
${SOFTKINETIC_SDK_LIB}
${OPENNI_LIB}
)

