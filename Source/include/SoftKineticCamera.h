#ifndef ___SOFTKINETICCAMERA_H_
#define ___SOFTKINETICCAMERA_H_

#include "CameraAbstract.h"
#include "DepthSense.hxx"
#include <tinythread.h>

using namespace DepthSense;

typedef struct{
uint8_t blue;
uint8_t green;
uint8_t red;
} bgr_pixel;


class SoftKineticCamera : public CameraAbstract {
private:
	void ConvertBGRToRGBA(const bgr_pixel* SoftKineticBGR, CAMERA_RGB_TYPE* d_newRgbMap);
	uint colorWidth,colorHeight;

	static CAMERA_DEPTH_TYPE* m_depthMap;
	static uint8_t* m_rgbMap;
	static bgr_pixel* m_cudaRGBMap;
	static CAMERA_DEPTH_TYPE* m_cudaDepthMap;
	static uint sizeOfDepthMap,sizeOfRGBMap;
	static bool m_bDeviceFound;
	static tthread::mutex depth_mutex;
	static tthread::mutex rgb_mutex;

	static void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data);
	static void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data);
	static void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data);
	static void configureAudioNode();
	static void configureDepthNode();
	static void configureColorNode();
	static void configureNode(Node node);
	static void onNodeConnected(Device device, Device::NodeAddedData data);
	static void onNodeDisconnected(Device device, Device::NodeRemovedData data);
	static void onDeviceConnected(DepthSense::Context context, DepthSense::Context::DeviceAddedData data);
	static void onDeviceDisconnected(DepthSense::Context context, DepthSense::Context::DeviceRemovedData data);
	
	tthread::thread* eventThread;

protected:

	
	virtual bool Init();
public:
		
	
	// C'tor.
	SoftKineticCamera();

	// D'tor.
	virtual ~SoftKineticCamera();

	// Get new depth frame.
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap);

	virtual char* GetCameraTypeStr() { return "SoftKineticCamera"; }
	// Get new rgb frame, if available.
	virtual bool GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap);


};





#endif // ___SOFTKINETICCAMERA_H_
