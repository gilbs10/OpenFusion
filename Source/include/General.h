#ifndef ___OUR_GENERAL____
#define ___OUR_GENERAL____

#include "utilities.h"

class General{
	static bool m_instanceFlag;
	static General* m_instance;

	General(){}
public:
	static General* Instance();
	~General(){
		m_instanceFlag = false;
	}


	void InitVoxelArray(VoxelType* d_voxel_array, VoxelWeightType* d_weight_array);
	void UpdateTransforms(float* d_currentTransform, float* d_invCurrentTransform, float3* d_cameraPosition, float* invViewMatrix);
};


#endif //___OUR_GENERAL____
