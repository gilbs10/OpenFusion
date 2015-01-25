#include "utilities.h"
#include "CudaUtilities.cuh"
#include "SoftKineticCamera.h"
//#include <iostream>	// TODO REMOVE
//using std::cout;

__global__ void d_ConvertBGRToRGBA(const bgr_pixel* SoftKineticBGR, CAMERA_RGB_TYPE* d_resultRgbMap,
									uint depthWidth,uint depthHeight,uint colorWidth,uint colorHeight) {
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= depthWidth) || (y >= depthHeight)) return;
	float widthRatio,heightRatio;
	widthRatio=colorWidth/depthWidth;
	heightRatio=colorHeight/depthHeight;

	uint offset = floor(heightRatio*(colorWidth)*y) + floor(widthRatio*x);
	uint8_t b = SoftKineticBGR[offset].blue;
	uint8_t g = SoftKineticBGR[offset].green;
	uint8_t r = SoftKineticBGR[offset].red;
	uint8_t a = 255;
	offset=depthWidth*y+x;
    d_resultRgbMap[offset] = (uint(a)<<24) | (uint(b)<<16) | (uint(g)<<8) | uint(r);
}

void SoftKineticCamera::ConvertBGRToRGBA(const bgr_pixel* SoftKineticBGR, CAMERA_RGB_TYPE* d_newRgbMap) {
	dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
	d_ConvertBGRToRGBA<<<gridSize, blockSize>>>(SoftKineticBGR, d_newRgbMap, m_params.m_width, m_params.m_height,colorWidth,colorHeight);
	cutilSafeCall(cudaDeviceSynchronize());
}

