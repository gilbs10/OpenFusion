#include "General.h"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>

#include <Eigen/Core>
#include <Eigen/Dense>

// Declarations for functions in General.cu
void initVoxelArrayCuda(dim3 dimGrid, dim3 dimBlock, VoxelType* d_voxel_array, VoxelWeightType* d_weight_array);
void updateTransformsCuda(float* d_currentTransform, float3* d_cameraPosition);
void addToCurrentTransformCuda(float* d_currentTransform, float* d_invViewMatrix);

void General::UpdateTransforms(float* d_currentTransform, float* d_invCurrentTransform, float3* d_cameraPosition, float* invViewMatrix)
{
	// If invViewMatrix == NULL then don't add the viewMatrix to the currentTransform
	if (invViewMatrix != NULL) {
		float* d_invViewMatrix = NULL;
		cutilSafeCall(cudaMalloc((void**)&d_invViewMatrix, 12*sizeof(float)));
		cutilSafeCall(cudaMemcpy(d_invViewMatrix, invViewMatrix, 12*sizeof(float), cudaMemcpyHostToDevice));
		addToCurrentTransformCuda(d_currentTransform, d_invViewMatrix);
		cutilSafeCall(cudaFree(d_invViewMatrix));
	}


	//Only d_currentTransform is up to date - we need to calculate and update the others.
	float invCurrentTransform_host[16];
	// copy current transform to host
	cutilSafeCall( cudaMemcpy(invCurrentTransform_host, d_currentTransform, 16*sizeof(float), cudaMemcpyDeviceToHost) );
	cudaDeviceSynchronize();
	// Calculate inverse of current transformation matrix.
	Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> > invCurrentTransformMatrix(invCurrentTransform_host);	// TODO: COLMAJOR IS ALSO WORKING WELL. CHECK OUT. MAYBE BUGGY
	invCurrentTransformMatrix = invCurrentTransformMatrix.inverse();

	// Copy current inverse transformation matrix back into cuda.
	cutilSafeCall( cudaMemcpy(d_invCurrentTransform, invCurrentTransform_host, 16*sizeof(float), cudaMemcpyHostToDevice) );
	cudaDeviceSynchronize();
	// At this point both d_currentTransform and d_invCurrentTransform are up to date.
	updateTransformsCuda(d_currentTransform, d_cameraPosition);
}

void General::InitVoxelArray(VoxelType* d_voxel_array, VoxelWeightType* d_weight_array)
{
	dim3 dimGrid(VOLUME_SIZE, VOLUME_SIZE);
	dim3 dimBlock(VOLUME_SIZE);
	initVoxelArrayCuda(dimGrid, dimBlock, d_voxel_array, d_weight_array);		// TODO: VOLUME_SIZE=512, AND THIS IS THE MAXIMUM THREADS FOR PARALLEL EXECUTION

	// TODO: THINK HOW TO DO THIS BETTER. DIVIDE THE THREADS INTO FEWER BLOCKS TO SAVE RESOURCES.
}

General* General::Instance() {
	if(!m_instanceFlag) {
		m_instance = new General();
		m_instanceFlag = true;
		return m_instance;
	} else {
		return m_instance;
	}
}
