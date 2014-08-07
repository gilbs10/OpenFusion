#ifndef ___CAMERAABSTRACT_H_
#define ___CAMERAABSTRACT_H_

#include "utilities.h"



class CameraAbstract {

protected:
	bool m_isInitialized;
	CameraParams m_params;
	CAMERA_DEPTH_TYPE* m_tempDepthArray;
	CAMERA_RGB_TYPE* m_tempRgbArray;
	
	// Calculates and sets camera's params. Not virtual.
	void SetParams(float foh, float fov, uint width, uint height);
	// Initializes camera, creates parameters struct.
	//Very Important!!!! CameraAbstract::Init() must be called from the end of each child class's Init() implementation.
	//It uses the cameraParams width and height which are only known at the end of the child's Init().
	virtual bool Init();

	void undistort(CameraParams m_params, 
				   CAMERA_DEPTH_TYPE* depthMap,  CAMERA_DEPTH_TYPE* newDepthMap);
	void AverageFilter(CAMERA_DEPTH_TYPE* d_newDepthMap);
	void GradientFilter(CAMERA_DEPTH_TYPE* d_newDepthMap);
	void MedianFilter(CAMERA_DEPTH_TYPE* d_newDepthMap);

public:

	int m_depthSizeBytes;
	int m_rgbSizeBytes;
	
	// C'tor.  Post: camera parameters are initialized.
	CameraAbstract();
	
	// Virtual destructor.
	virtual ~CameraAbstract();
	
	// Getter.
	bool IsInitialized();
	
	// Getter.
	CameraParams* GetParameters();

	// Get new depth frame.
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) = 0;

	virtual char* GetCameraTypeStr() = 0;

	// Get new rgb frame, if available.
	virtual bool GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
		return false;
	};

	// TODO: Add functions readToFile, writeToFile in the abstract class. // notice that we need to add a configurations file for each camera when it writes to file.

};

#endif // ___CAMERAABSTRACT_H_

