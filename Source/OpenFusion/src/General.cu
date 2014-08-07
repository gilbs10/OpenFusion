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

// Simple 3D volume renderer


#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>

#include "General.h"
#include <math.h>

///////////// General functions: ///////////

bool General::m_instanceFlag = false;
General* General::m_instance = NULL;


__global__ void d_initVoxelArray(VoxelType* d_voxel_array, VoxelWeightType* d_weight_array)
{
	uint x_index = blockIdx.x; //blockIdx.x * blockDim.x + threadIdx.x;
    uint y_index = blockIdx.y; //blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((x_index >= VOLUME_SIZE) || (y_index >= VOLUME_SIZE)) return;

	uint offset = x_index*VOLUME_SIZE*VOLUME_SIZE + y_index*VOLUME_SIZE;
	uint z_index = threadIdx.x; //blockIdx.x + offset;

	//float x = (float)x_index;
	//float y = (float)y_index;
	//float z = (float)z_index;

	offset += z_index;

	// At the end: weight should be zero. tsdf should be 1.

	d_voxel_array[offset].tsdf = TRUNCATION;
	d_weight_array[offset] = 0;

	return;

}

//   Rotation = { 0  1  2        Transform = { 0   1   2   3            Transform = {0   4   8   12      
//                3  4  5                      4   5   6   7            Transposed   1   5   9   13
//                6  7  8}                     8   9   10  11                        2   6   10  14  
//                                             12  13  14  15}                       3   7   11  15
// Transposes the rotation matrix of currentTransform, and extract the camera position.
__global__ void d_updateTransforms (float* d_currentTransform, float3* d_cameraPosition)
{	
	d_cameraPosition->x = d_currentTransform[3];
	d_cameraPosition->y = d_currentTransform[7];
	d_cameraPosition->z = d_currentTransform[11];
}

__global__ void d_addToCurrentTransform(float* d_currentTransform, float* d_invViewMatrix) {
	float result[12] = {0.f};
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			for (int k = 0; k < 4; ++k) {
				result[i * 4 + j] += d_invViewMatrix[i * 4 + k] * d_currentTransform[k * 4 + j];
			}
		}
	}
	for (int i = 0; i < 12; ++i) {	// The last row of currentTransform remains (0,0,0,1)
		d_currentTransform[i] = result[i];
	}
}


///////////////////////// C++ functions ////////////////////////////////////////////////////

void initVoxelArrayCuda(dim3 dimGrid, dim3 dimBlock, VoxelType* d_voxel_array, VoxelWeightType* d_weight_array) {
	d_initVoxelArray<<<dimGrid, dimBlock>>>(d_voxel_array, d_weight_array);
}

void updateTransformsCuda(float* d_currentTransform, float3* d_cameraPosition) {
	d_updateTransforms<<<1, 1>>>(d_currentTransform, d_cameraPosition);	
}

void addToCurrentTransformCuda(float* d_currentTransform, float* d_invViewMatrix) {
	d_addToCurrentTransform<<<1,1>>>(d_currentTransform, d_invViewMatrix);
}


// MOVED TO General.cpp