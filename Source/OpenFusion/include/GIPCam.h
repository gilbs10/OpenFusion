#ifndef ___GIPCAM_H_
#define ___GIPCAM_H_

#include "CameraAbstract.h"

class GIPCam : public CameraAbstract {
protected:
	// private stuff

	virtual bool Init();

	void ConvertDepthMap(float* d_data, CAMERA_DEPTH_TYPE* d_newDepthMap, uint width, uint height, float z_offset);
	//void ReduceNoise(CAMERA_DEPTH_TYPE* d_newDepthMap, uint width, uint height); //TODO - WAS MOVED TO CAMERA ABSTRACT
	float* m_zCoordsMatrix; // holds the z values of the incoming frames (the incoming depth).  Before conversion to camera depth type
	int m_zCoordsMatrixSize;

	int m_currentFrameNumber;

public:
	// C'tor.
	GIPCam();

	// D'tor.
	virtual ~GIPCam();

	// Set parameters.
	void SetParams(float fx, float fy, float cx, float cy, uint width, uint height);

	// Get new depth frame.
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap);

	virtual char* GetCameraTypeStr() { return "GIPCam"; }

	// Get new rgb frame, if available.
	//virtual void GetRgbFrame(); // TODO
};


#endif // ___GIPCAM_H_
