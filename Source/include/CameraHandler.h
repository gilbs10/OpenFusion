#ifndef ___CAMERAHANDLER_H_
#define ___CAMERAHANDLER_H_

#include "utilities.h"
#include "CameraAbstract.h"

//using namespace cv;

class CameraHandler {

	CameraAbstract* m_camera;
	static CameraParams m_params; // TODO: WHY STATIC???? WHAT ABOUT RESET?

	CAMERA_DEPTH_TYPE* inputData_depth;
	CAMERA_RGB_TYPE* inputData_rgb;
	void writeInputToFile();
	void readInputFromFile(CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_RGB_TYPE* newRgbMap);
	void readInputFromCamera(CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_RGB_TYPE* newRgbMap);
public:
	CameraHandler();
	~CameraHandler();
	CameraParams* GetCameraParams();
	void cameraIteration(CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_RGB_TYPE* newRgbMap, uint* newDepthMap, const CameraData& d_cameraData, float* d_currentTransform, uint* newNormalMap, dim3 gridSize, dim3 blockSize); // TODO - uppercase first letter
	static void updateCameraData(dim3 gridSize, dim3 blockSize, const CameraData& d_cameraData, CAMERA_DEPTH_TYPE* d_newDepth, float* d_currentTransform, uint* newNormalMap, uint* newDepthMap);
	void SwitchToWriteMode();
	void SwitchToReadMode();
};

inline CameraParams* CameraHandler::GetCameraParams() {
	return m_camera->GetParameters();
}

#endif // ___CAMERAHANDLER_H_
