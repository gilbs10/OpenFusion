#ifndef ___RAYCASTER_H_
#define ___RAYCASTER_H_

#include "BasicTypes.h"

class Raycaster {
	dim3 m_numBlocks;
	CameraParams m_cameraParams;

public:
	Raycaster(CameraParams* cameraParams);
	void Raycast(uint *d_output, float* transform, VoxelType* d_voxel_array, const PointInfo& _renderedVertexMap, const float3* d_cameraPosition, bool showNormals);
};


#endif // ___RAYCASTER_H_
