#include "utilities.h"
#include "CudaUtilities.cuh"
#include "PrimesenseCamera.h"

__global__ void d_ConvertRGBToRGBA(const XnRGB24Pixel* primesense_rgb, CAMERA_RGB_TYPE* d_resultRgbMap, uint width, uint height) {
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= width) || (y >= height)) return;
	const uint offset = width*y + x;

	XnUInt8 r = primesense_rgb[offset].nRed;
	XnUInt8 g = primesense_rgb[offset].nGreen;
	XnUInt8 b = primesense_rgb[offset].nBlue;
	XnUInt8 a = 255;

    d_resultRgbMap[offset] = (uint(a)<<24) | (uint(b)<<16) | (uint(g)<<8) | uint(r);
}


void PrimesenseCamera::ConvertRGBToRGBA(const XnRGB24Pixel* primesense_rgb, CAMERA_RGB_TYPE* d_newRgbMap) {
	dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
	d_ConvertRGBToRGBA<<<gridSize, blockSize>>>(primesense_rgb, d_newRgbMap, m_params.m_width, m_params.m_height);
	cutilSafeCall(cudaDeviceSynchronize());
}