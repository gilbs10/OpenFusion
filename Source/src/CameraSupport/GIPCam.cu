#include "GIPCam.h"
#include "utilities.h"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h> // TODO: PROBABLY REDUNDANT

// The conversion has two purposes: 1. Convert from decimeters to mm by multiplying Z-coordinate by 100.
//									2. Convert from float to CAMERA_DEPTH_TYPE.
__global__ void d_ConvertDepthMap(float* d_data, CAMERA_DEPTH_TYPE* d_newDepthMap, uint width, uint height, float z_offset) {
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= width) || (y >= height)) return;
	const uint offset_camera = width*y + x;
	const uint offset_data = height*(width-1-x) + y;

	// Converting from decimeters to mm, plus translating so the camera is at the origin.
	d_newDepthMap[offset_camera] = (CAMERA_DEPTH_TYPE)(d_data[offset_data] * 100.f + z_offset); // TODO: Z is taken from the LOG file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MAY BE CHANGED IN CALIBRATION
}

void GIPCam::ConvertDepthMap(float* d_data, CAMERA_DEPTH_TYPE* d_newDepthMap, uint width, uint height, float z_offset) {
	dim3 gridSize(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y)); // TODO - MAKE THIS GRID_DIM A PARAMETER OF CAMERA_ABSTRACT. IT SHOULD BE THE RESOLUTION OF THE PICTURE. UPDATE_CAMERA_DATA SHOULD USE IT TOO
	d_ConvertDepthMap<<<gridSize, blockSize>>>(d_data, d_newDepthMap, width, height, z_offset);
	cutilSafeCall(cudaDeviceSynchronize());
}
