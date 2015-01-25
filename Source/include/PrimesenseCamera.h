#ifndef ___PRIMESENSECAMERA_H_
#define ___PRIMESENSECAMERA_H_

#include "CameraAbstract.h"
#include <XnOpenNI.h>
#include <XnCppWrapper.h>

using xn::Context;
using xn::ScriptNode;
using xn::DepthGenerator;
using xn::DepthMetaData;
using xn::ImageGenerator;
using xn::ImageMetaData;

class PrimesenseCamera : public CameraAbstract {
private:
	void ConvertRGBToRGBA(const XnRGB24Pixel* primesense_rgb, CAMERA_RGB_TYPE* d_newRgbMap);
protected:
	Context m_context;
	ScriptNode m_scriptNode;
	DepthGenerator m_depth;
	DepthMetaData m_depthMD;
	ImageGenerator m_rgb;
	ImageMetaData m_rgbMD;
	XnRGB24Pixel* md_source_rgb; // Used for storing the source rgb image, before converting to rgba.

	virtual bool Init();
public:
	// C'tor.
	PrimesenseCamera();

	// D'tor.
	virtual ~PrimesenseCamera();

	// Get new depth frame.
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap);

	virtual char* GetCameraTypeStr() { return "PrimesenseCamera"; }
	// Get new rgb frame, if available.
	virtual bool GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap);
};

#endif // ___PRIMESENSECAMERA_H_
