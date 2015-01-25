#ifndef ___CAMERAFROMFILE_H_
#define ___CAMERAFROMFILE_H_

#include "CameraAbstract.h"

class CameraFromFile : public CameraAbstract {
protected:
	char m_path[200];
	int m_currentFrameNumber;
	CAMERA_DEPTH_TYPE* m_inputDataDepth;
	CAMERA_RGB_TYPE* m_inputDataRgb;

	virtual bool Init();
public:
	// C'tor.
	CameraFromFile(char* path);

	// D'tor.
	virtual ~CameraFromFile();

	// Get new depth frame.
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap);
	
	// Get new rgb frame, if available.
	virtual bool GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap);

	virtual char* GetCameraTypeStr() { return "CameraFromFile"; }
};

#endif // ___CAMERAFROMFILE_H_
