#include "CameraAbstract.h"

// CUDA utilities and system includes
#include "cutil_inline.h"    // includes cuda.h and cuda_runtime_api.h

#include <iostream> // TODO REMOVE
using namespace std;

void CameraAbstract::SetParams(float foh, float fov, uint width, uint height) {

	cout << "FOH: " << foh << endl;
	cout << "FOV: " << fov << endl;

	///////////////////////////////////////////////////////////////////////////////
	///////////////// TODO WANTED!!!!!!!!!!!!!!     ///////////////////////////////
//	foh = 1.01229096; - 58 degrees in radian
//	fov = 0.78539816; - 45 degrees in radian
	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////


	// TODO, using fov, foh from camera, maybe problematic (slightly different angles than what we're used to).


	float focalLengthX = tan(foh*0.5)*2.f; // Horizontal focal length (foh in radians).
	float focalLengthY = tan(fov*0.5)*2.f; // Vertical focal length (fov in radians).

	// Intrinsic.
	m_params.m_intrinsic.Set(  -(float)width / focalLengthX,
							   -(float)height / focalLengthY,
								width * 0.5,
								height * 0.5);

	// Inverse intrinsic.
	m_params.m_invIntrinsic.Set(   -focalLengthX / (float)width,
								   -focalLengthY / (float)height,
									focalLengthX * 0.5,
									focalLengthY * 0.5);

	m_params.m_width = width;
	m_params.m_height = height;
}

CameraAbstract::CameraAbstract() : m_isInitialized(false), 
								   m_tempDepthArray(NULL), 
								   m_depthSizeBytes(0), 
								   m_rgbSizeBytes(0) {
	
}

CameraAbstract::~CameraAbstract() {
	cutilSafeCall(cudaFree(m_tempDepthArray));
}
	
bool CameraAbstract::IsInitialized() { 
	return m_isInitialized; 
}

CameraParams* CameraAbstract::GetParameters() { 
	return &m_params;
}


// Note that this must be called at the end of the child's Init() implementation
// (because only then do we know the width/height)
bool CameraAbstract::Init() {
	m_depthSizeBytes =  m_params.m_width*m_params.m_height*sizeof(CAMERA_DEPTH_TYPE);
	cutilSafeCall(cudaMalloc((void**)&m_tempDepthArray, m_depthSizeBytes));
	cutilSafeCall(cudaMemset(m_tempDepthArray, 0, m_depthSizeBytes));

	m_rgbSizeBytes =  m_params.m_width*m_params.m_height*sizeof(CAMERA_RGB_TYPE);
	cutilSafeCall(cudaMalloc((void**)&m_tempRgbArray, m_rgbSizeBytes));
	cutilSafeCall(cudaMemset(m_tempRgbArray, 0, m_rgbSizeBytes));

	return true;
}
