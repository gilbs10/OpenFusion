#ifndef ___ICP_H_
#define ___ICP_H_

#include "utilities.h"

#define NUM_VARS (6)

class ICP {
	float* d_A_Matrix;				// matrix of dim: (imageW*ImageH)x6
	float* d_b_Matrix;				// array of dim: (imageW*ImageH)x1

	float* AtA;						// matrix of dim: (36)xCorrespondence_block
	float* Atb;						// array of dim: Correspondence_block

	float* Atransposed;				// matrix of dim: 6x(imageW*ImageH)
	float* Atransposed_x_A;			// The multiplication of Atransposed by A (a 6x6 matrix)
	float* inv_Atransposed_x_A;		// The inverse of Atransposed by A (a 6x6 matrix)
	float* Atransposed_x_b;			// The multiplication of Atransposed by b (a 6x1 matrix)
	float* xOpt;					// The values of alpha, beta, gamma, tx,ty, tz, which make up the new transormation matrix.
									// (a 6x1 matrix)
	float* invLastTransform;		// pointer to previous transformation matrix in cuda
	float* newTransform;			// pointer to current (new) transformation matrix in cuda
	float* incrementalTransform;	// pointer to the incremental transform matrix created for this frame in all ICP iterations
	float* bestTransform;			// pointer to the best transform matrix found so far (minimum average error)
	float* errorSum;				// single float, holds sum of errors calculated for new transformation matrix.

	float* Atransposed_x_A_host;
	float* invLastTransform_host;	// the inverse of the transformation from the previous iteration of ICP

	//TODO - can these be removed entirely?  They're only needed for the call to camera handler for updateCameraData
	dim3 m_gridSize; //The grid size for GPU calls
	dim3 m_blockSize; //The block size for GPU calls

	const int m_maxIterations; //number of icp iterations

	void Cpu_buildNewTransformMatrix (float* xOpt, float* newTransform, float* fullIncrementalTransform); //builds the new transformation matrix
	void ClearHostMatrices();

	// host matrices
	float AtA_sum[NUM_VARS*NUM_VARS];
	float Atb_sum[NUM_VARS];
	float* AtA_host;
	float* Atb_host;
	float* xOpt_host;
	CameraParams m_cameraParams;

	uint m_numCorrespondenceBlocks;
	

public:

	//called once at the beginning of the program, allocates A,b arrays in cuda
	ICP (dim3 gridSize, dim3 blockSize, CameraParams* cameraParams);
	~ICP ();

	//receives depth map and vertex map pointers, previous transformation matrix, performs ICP
	// The output is in @Param _newTransform_output
	void Iterate_ICP (const CameraData& _newDepthData,
					  const PointInfo& _currentVertexMap,
					  float* _currentTransform);
	
	void FindCorresponding(const CameraData& newDepthMap, const PointInfo& currentVertexMap);

	void BuildNewTransformMatrix ();

	void CopyProjectionMatrix(const float *_kMat, const float *_kInvMat);

	void CopyWidthAndHeight(const int _width, const int _height);

};

#endif // #ifndef ___ICP_H_