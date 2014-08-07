#ifndef ___WRITETOFILE_H_
#define ___WRITETOFILE_H_

#include "CameraAbstract.h"
#include <fstream> // TODO DON'T REMOVE
using std::ofstream;

class WriteToFile : public CameraAbstract {
protected:
	CameraAbstract* m_camera;

	char m_dirName[100]; //directory for write
	ofstream m_settingsFile;
	uint m_frameNumber;
	CAMERA_DEPTH_TYPE* m_inputDataDepth;
	CAMERA_RGB_TYPE* m_inputDataRgb;

	virtual bool Init();
public:
	// C'tor.
	WriteToFile(CameraAbstract* camera);

	// D'tor.
	virtual ~WriteToFile();

	CameraParams* GetParameters();

	// Get new depth frame.
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap);
	
	// Get new rgb frame, if available.
	virtual bool GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap);
	
	virtual char* GetCameraTypeStr() { return "WriteToFile"; }
};

#endif // ___WRITETOFILE_H_
