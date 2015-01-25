#cam



include_directories (
${GLEW_INCLUDE_DIR}
"${PROJECT_SOURCE_DIR}/include"
"${PROJECT_SOURCE_DIR}/include/TinyThread"
"${PROJECT_SOURCE_DIR}/include/Eigen"
"${PROJECT_SOURCE_DIR}/include/Utils"
${CUDA_TOOLKIT_INCLUDE}
${OPENNI_INCLUDE}
${SOFTKINETIC_SDK_INCLUDE})

set(SRCSCU CameraAbstract.cu CameraHandler.cu)
set(SRCS CameraAbstract.cpp CameraHandler.cpp WriteToFile.cpp)

if(BUILD_FROMFILE) 
   MESSAGE(STATUS "build fromfile")
   list(APPEND SRCS CameraFromFile.cpp)
endif()
if(BUILD_SOFTKINETIC) 
   MESSAGE(STATUS "build softkinetic")
   list(APPEND SRCS SoftKineticCamera.cpp)
   list(APPEND SRCSCU SoftKineticCamera.cu)
endif()
if(BUILD_PRIMESENSE) 
   MESSAGE(STATUS "build primesense")
   list(APPEND SRCS PrimesenseCamera.cpp)
   list(APPEND SRCSCU PrimesenseCamera.cu)
endif()
if(BUILD_GIPCAM) 
   MESSAGE(STATUS "build gipcam")
   list(APPEND SRCS GIPCam.cpp)
   list(APPEND SRCSCU GIPCam.cu)
endif()
if (NOT BUILD_SOFTKINETIC AND NOT BUILD_PRIMESENSE AND NOT BUILD_GIPCAM AND NOT BUILD_FROMFILE)
  MESSAGE(FATAL_ERROR "no camera")
endif()

cuda_add_library(OpenFusionCamCU ${SRCSCU})
add_library(OpenFusionCam ${SRCS})
