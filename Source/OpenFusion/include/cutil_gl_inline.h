/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#ifndef _CUTIL_GL_INLINE_H_
#define _CUTIL_GL_INLINE_H_

//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>

#include <cuda.h>
#include <cutil.h>
//#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


#if __DEVICE_EMULATION__
    inline int cutilGLDeviceInit(int ARGC, char **ARGV) { return 0; }
    inline int cutilGLDeviceInitDrv(int cuDevice, int ARGC, char **ARGV) { return 0; } 
    inline void cutilChooseCudaGLDevice(int ARGC, char **ARGV) { }
#else
    inline int cutilGLDeviceInit(int ARGC, char **ARGV)
    {
        int deviceCount;
        cutilSafeCallNoSync(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "CUTIL CUDA error: no devices supporting CUDA.\n");
            exit(-1);
        }
        int dev = 0;
        cudaDeviceProp deviceProp;
        cutilSafeCallNoSync(cudaGetDeviceProperties(&deviceProp, dev));
        if (deviceProp.major < 1) {
            fprintf(stderr, "cutil error: device does not support CUDA.\n");
            exit(-1);
        }
        cutilSafeCall(cudaGLSetGLDevice(dev));
        return dev;
    }

    // This function will pick the best CUDA device available with OpenGL interop
    inline int cutilChooseCudaGLDevice()
    {
		// Pick the device with highest Gflops/s
		int devID = cutGetMaxGflopsDeviceId();
        cudaGLSetGLDevice( devID );
		return devID;
    }

#endif

#endif // _CUTIL_GL_INLINE_H_
