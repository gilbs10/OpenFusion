#include <iostream>	// TODO REMOVE
#include "PrimesenseCamera.h"
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h // TODO REMOVE - NOT MODULAR
#include "CudaUtilities.cuh"

#include <Eigen/Core>
#include <Eigen/LU>

#define SAMPLE_XML_PATH "data\\OpenNIConfig.xml" // TODO: Use FileSystemHelper.h to build path

#define PRIMESENSE_MIN_DEPTH_MM (300.f)
#define PRIMESENSE_MAX_DEPTH_MM (6000.f)

using std::cout;
using std::endl;
using xn::EnumerationErrors;

bool CheckReturnValue (const XnStatus rc, const char* what) {	// TODO THIS IS CALLED JUST ONCE (IF THERE IS NO RGB)? MAYBE REDUNDANT.
	if (rc != XN_STATUS_OK)	{
		cout << what << " failed: " << xnGetStatusString(rc) << endl;
		return false;
	}
	return true;
}

PrimesenseCamera::PrimesenseCamera() {
	m_isInitialized = Init();
}

PrimesenseCamera::~PrimesenseCamera() {
	cutilSafeCall(cudaFree(md_source_rgb));
	m_depth.Release();
	m_rgb.Release();
	m_scriptNode.Release();
	m_context.Release();
}

bool PrimesenseCamera::Init() {
	XnStatus nRetVal = XN_STATUS_OK;

	EnumerationErrors errors;

	nRetVal = m_context.InitFromXmlFile(SAMPLE_XML_PATH, m_scriptNode, &errors);
	//m_context.SetGlobalMirror(false);

	if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
	{
		XnChar strError[1024];
		errors.ToString(strError, 1024);
		cout << strError << endl;
		return false;
	}
	else if (nRetVal != XN_STATUS_OK)
	{
		cout << "Open failed: " << xnGetStatusString(nRetVal) << endl;
		return false;
	}

	nRetVal = m_context.FindExistingNode(XN_NODE_TYPE_DEPTH, m_depth);
	if (!CheckReturnValue(nRetVal, "Find depth generator")) {
		return false;
	} // TODO CONSIDER REFACTOR ALL THESE IF'S WITH ERRORS

	nRetVal = m_context.FindExistingNode(XN_NODE_TYPE_IMAGE, m_rgb);
	if (!CheckReturnValue(nRetVal, "Find rgb generator")) {
		return false;
	} // TODO CONSIDER REFACTOR ALL THESE IF'S WITH ERRORS

	
	XnMapOutputMode rgb_mode;
    rgb_mode.nXRes = XN_VGA_X_RES;
    rgb_mode.nYRes = XN_VGA_Y_RES;
    rgb_mode.nFPS = 30;
	XnStatus status = m_rgb.SetMapOutputMode(rgb_mode);
    if(status != XN_STATUS_OK) {
		cout << "Setting output mode to RGB node failed." << endl;
		return false;
	}
	

	//if(deviceType == DEVICE_ASUS_XTION) // TODO: Those are ASUS-specific. Might not fit to Kinect!!!!!
    //{
        //ps/asus specific
		m_rgb.SetIntProperty("InputFormat", 1 /*XN_IO_IMAGE_FORMAT_YUV422*/);
		m_rgb.SetPixelFormat(XN_PIXEL_FORMAT_RGB24);
		m_depth.SetIntProperty("RegistrationType", 1 /*XN_PROCESSING_HARDWARE*/);
	//}

	// Get parameters of image.
	m_depth.GetMetaData(m_depthMD); // TODO - PAY ATTENTION!!! THE FIRST FRAME IS BEING DEPRECATED!
	m_rgb.GetMetaData(m_rgbMD);
	XnFieldOfView fov;
	m_depth.GetFieldOfView(fov);

	SetParams(fov.fHFOV, fov.fVFOV, m_depthMD.XRes(), m_depthMD.YRes());

	m_params.m_z_offset = 0.f;
	m_params.m_min_depth = PRIMESENSE_MIN_DEPTH_MM;
	m_params.m_max_depth = PRIMESENSE_MAX_DEPTH_MM;

	cutilSafeCall(cudaMalloc((void**)&md_source_rgb, m_params.m_width*m_params.m_height*sizeof(XnRGB24Pixel)));
	
	CameraAbstract::Init();

	return true;
}


	////////////////////////////////////////////////////////////// PROBLEM //////////////////////////////
	// TAKE A LOOK AT THIS:
	//
/*			bool modeRes = false;
		modeRes = capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ );
		capture.set( CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION, 0 );

		cout << capture.get(CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION) << endl;

		*/

	// TRY TO LOOK AT OPENCV IMPLEMENTATION

bool PrimesenseCamera::GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	XnStatus nRetVal;

	nRetVal = m_context.WaitOneUpdateAll(m_depth);
	if (nRetVal != XN_STATUS_OK)
	{
		printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
		return false;
	}

	m_depth.GetMetaData(m_depthMD);
	
	if (d_newDepthMap) {
		cutilSafeCall(cudaMemcpy(d_newDepthMap, m_depthMD.Data(), m_params.m_width*m_params.m_height*sizeof(CAMERA_DEPTH_TYPE), cudaMemcpyHostToDevice)); 
	}

	return true;
}

bool PrimesenseCamera::GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
	XnStatus nRetVal;

	nRetVal = m_context.WaitOneUpdateAll(m_rgb);
	if (nRetVal != XN_STATUS_OK)
	{
		cout << "UpdateData Rgb failed: " << xnGetStatusString(nRetVal) << endl;
		return false;
	}

	m_rgb.GetMetaData(m_rgbMD);

	if (!m_rgbMD.Data()) {
		return false;
	}

	if(m_rgbMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24) {
        cout << "Unsupported format of grabbed rgb image." << endl;
		return false;
	}

	if (d_newRgbMap) {
		cutilSafeCall(cudaMemcpy(md_source_rgb, m_rgbMD.RGB24Data(), m_params.m_width*m_params.m_height*sizeof(XnRGB24Pixel), cudaMemcpyHostToDevice));
		ConvertRGBToRGBA(md_source_rgb, d_newRgbMap);
	}

	return true;
}
