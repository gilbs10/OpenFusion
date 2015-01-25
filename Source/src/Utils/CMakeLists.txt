#utils

include_directories (
${GLEW_INCLUDE_DIR}
"${PROJECT_SOURCE_DIR}/include"
"${PROJECT_SOURCE_DIR}/include/TinyThread"
"${PROJECT_SOURCE_DIR}/include/Eigen"
"${PROJECT_SOURCE_DIR}/include/Utils"
${CUDA_TOOLKIT_INCLUDE}
${OPEN_NI_INCLUDE}
${SOFTKINETIC_SDK_INCLUDE})


set(SRCS cutil.cpp  FileSystemHelper.cpp stopwatch.cpp tinythread.cpp myUtils.cpp)
if(UNIX) 
  list(APPEND SRCS stopwatch_linux.cpp)
else()
  list(APPEND SRCS stopwatch_win.cpp)
endif()

add_library(OpenFusionUtils ${SRCS})
