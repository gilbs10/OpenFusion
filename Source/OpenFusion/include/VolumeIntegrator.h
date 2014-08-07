#pragma once

#ifndef ___TSDF_H_
#define ___TSDF_H_

#include "utilities.h"


class VolumeIntegrator{
	const CameraParams m_cameraParams;
	dim3 m_numBlocks;

public:
	VolumeIntegrator(CameraParams* cameraParams);
	void Integrate (VoxelType* d_voxel_array, VoxelWeightType* d_weight_array, const float* const invTransform, const CameraData& depthMap, const float3* d_cameraPosition);

};



#endif // ___TSDF_H_
