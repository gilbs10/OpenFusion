#src


include_directories (
${GLEW_INCLUDE_DIR}
"${PROJECT_SOURCE_DIR}/include"
"${PROJECT_SOURCE_DIR}/include/TinyThread"
"${PROJECT_SOURCE_DIR}/include/Eigen"
"${PROJECT_SOURCE_DIR}/include/Utils"
${CUDA_TOOLKIT_INCLUDE}
${OPENNI_INCLUDE}
${SOFTKINETIC_SDK_INCLUDE})


set(SRCSCU Raycaster.cu General.cu ICP.cu VolumeIntegrator.cu)
set(SRCS General.cpp  ICP.cpp)

add_subdirectory(Utils)
add_subdirectory(CameraSupport)
cuda_add_library(OpenFusionSrcCU ${SRCSCU})
add_library(OpenFusionSrc ${SRCS})

