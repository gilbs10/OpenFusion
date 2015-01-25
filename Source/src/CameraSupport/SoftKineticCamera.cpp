


#include <iostream>	// TODO REMOVE
#include "SoftKineticCamera.h"
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h // TODO REMOVE - NOT MODULAR
#include "CudaUtilities.cuh"

#include <tinythread.h>
//#include <vector>
#include <exception>



using std::cout;
using std::endl;
using std::vector;




//TODO: verify
#define SOFTKINETIC_MIN_DEPTH_MM (150.f)
#define SOFTKINETIC_MAX_DEPTH_MM (1000.f)

//Those are the depth fov parameters the color sensor are slightly different
//#define SOFTKINETIC_FOV_RADIANS 1.012290966
//#define SOFTKINETIC_FOH_RADIANS 1.291543646 
#define SOFTKINETIC_FOV_RADIANS 0.960015866
#define SOFTKINETIC_FOH_RADIANS 1.238382182
 

#define SOFTKINETIC_K1 -0.17010300
#define SOFTKINETIC_K2 0.14406399
#define SOFTKINETIC_P1 0
#define SOFTKINETIC_P2 0
#define SOFTKINETIC_DEPTH_NULL_VALUE 32001


Context m_context;
DepthNode m_dnode;
ColorNode m_cnode;
AudioNode m_anode;
Device m_device;

tthread::mutex  SoftKineticCamera::depth_mutex;
tthread::mutex  SoftKineticCamera::rgb_mutex;

CAMERA_DEPTH_TYPE* SoftKineticCamera::m_depthMap=NULL;
uint8_t* SoftKineticCamera::m_rgbMap=NULL;
bgr_pixel* SoftKineticCamera::m_cudaRGBMap=NULL;
CAMERA_DEPTH_TYPE* SoftKineticCamera::m_cudaDepthMap=NULL;
uint SoftKineticCamera::sizeOfDepthMap;
uint SoftKineticCamera::sizeOfRGBMap;
bool SoftKineticCamera::m_bDeviceFound=false;

/*
###########################################################################
####################### Softkinetic event handlers ########################
###########################################################################
*/

/*----------------------------------------------------------------------------*/
// New audio sample event handler
void SoftKineticCamera::onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
	//Do nothing
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
void SoftKineticCamera::onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{	
	if (data.colorMap) {
		tthread::lock_guard<tthread::mutex> guard(rgb_mutex);
		memcpy(m_rgbMap, data.colorMap, sizeOfRGBMap);
	}
}

/*----------------------------------------------------------------------------*/
// New depth sample event handler
void SoftKineticCamera::onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
	if(data.depthMap) {
		tthread::lock_guard<tthread::mutex> g(depth_mutex);
		memcpy(m_depthMap, data.depthMap, sizeOfDepthMap);
	}
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::configureAudioNode()
{
	m_anode.newSampleReceivedEvent().connect(&onNewAudioSample);

	AudioNode::Configuration config = m_anode.getConfiguration();
	config.sampleRate = 44100;

	try 
	{
		m_context.requestControl(m_anode,0);

		m_anode.setConfiguration(config);

		m_anode.setInputMixerLevel(0.5f);
	}
	catch (ArgumentException& e)
	{
		printf("Argument Exception: %s\n",e.what());
	}
	catch (UnauthorizedAccessException& e)
	{
		printf("Unauthorized Access Exception: %s\n",e.what());
	}
	catch (ConfigurationException& e)
	{
		printf("Configuration Exception: %s\n",e.what());
	}
	catch (StreamingException& e)
	{
		printf("Streaming Exception: %s\n",e.what());
	}
	catch (TimeoutException&)
	{
		printf("TimeoutException\n");
	}
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::configureDepthNode()
{
	m_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);

	DepthNode::Configuration config = m_dnode.getConfiguration();
	config.frameFormat = FRAME_FORMAT_QVGA;
	config.framerate = 25;
	config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
	config.saturation = true;

	m_dnode.setEnableDepthMap(true);
	try 
	{
		m_context.requestControl(m_dnode,0);
		m_dnode.setEnableConfidenceMap(true);
		m_dnode.setConfidenceThreshold(120);
		m_dnode.setConfiguration(config);
	}
	catch (ArgumentException& e)
	{
		printf("Argument Exception: %s\n",e.what());
	}
	catch (UnauthorizedAccessException& e)
	{
		printf("Unauthorized Access Exception: %s\n",e.what());
	}
	catch (IOException& e)
	{
		printf("IO Exception: %s\n",e.what());
	}
	catch (InvalidOperationException& e)
	{
		printf("Invalid Operation Exception: %s\n",e.what());
	}
	catch (ConfigurationException& e)
	{
		printf("Configuration Exception: %s\n",e.what());
	}
	catch (StreamingException& e)
	{
		printf("Streaming Exception: %s\n",e.what());
	}
	catch (TimeoutException&)
	{
		printf("TimeoutException\n");
	}

}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::configureColorNode()
{
	// connect new color sample handler
	m_cnode.newSampleReceivedEvent().connect(&onNewColorSample);

	ColorNode::Configuration config = m_cnode.getConfiguration();
	config.frameFormat = FRAME_FORMAT_VGA;
	config.compression = COMPRESSION_TYPE_MJPEG;
	config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
	config.framerate = 25;



	m_cnode.setEnableColorMap(true);

	try 
	{
		m_context.requestControl(m_cnode,0);

		m_cnode.setConfiguration(config);
	}
	catch (ArgumentException& e)
	{
		printf("Argument Exception: %s\n",e.what());
	}
	catch (UnauthorizedAccessException& e)
	{
		printf("Unauthorized Access Exception: %s\n",e.what());
	}
	catch (IOException& e)
	{
		printf("IO Exception: %s\n",e.what());
	}
	catch (InvalidOperationException& e)
	{
		printf("Invalid Operation Exception: %s\n",e.what());
	}
	catch (ConfigurationException& e)
	{
		printf("Configuration Exception: %s\n",e.what());
	}
	catch (StreamingException& e)
	{
		printf("Streaming Exception: %s\n",e.what());
	}
	catch (TimeoutException&)
	{
		printf("TimeoutException\n");
	}
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::configureNode(Node node)
{
	if ((node.is<DepthNode>())&&(!m_dnode.isSet()))
	{
		m_dnode = node.as<DepthNode>();
		configureDepthNode();
		m_context.registerNode(node);
	}

	if ((node.is<ColorNode>())&&(!m_cnode.isSet()))
	{
		m_cnode = node.as<ColorNode>();
		configureColorNode();
		m_context.registerNode(node);
	}

	if ((node.is<AudioNode>())&&(!m_anode.isSet()))
	{
		m_anode = node.as<AudioNode>();
		configureAudioNode();
		m_context.registerNode(node);
	}
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::onNodeConnected(Device device, Device::NodeAddedData data)
{
	configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
	if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == m_anode))
		m_anode.unset();
	if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == m_cnode))
		m_cnode.unset();
	if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == m_dnode))
		m_dnode.unset();
	printf("Node disconnected\n");
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::onDeviceConnected(DepthSense::Context context, DepthSense::Context::DeviceAddedData data)
{
	if (!m_bDeviceFound)
	{
		data.device.nodeAddedEvent().connect(&SoftKineticCamera::onNodeConnected);
		data.device.nodeRemovedEvent().connect(&SoftKineticCamera::onNodeDisconnected);
		m_bDeviceFound = true;
	}
}

/*----------------------------------------------------------------------------*/
void SoftKineticCamera::onDeviceDisconnected(DepthSense::Context context, DepthSense::Context::DeviceRemovedData data)
{
	m_bDeviceFound = false;
	printf("Device disconnected\n");
}


void runEvents(void* arg) {
	try 
	{
		m_context.run();
	}
	catch (ArgumentException& e)
	{
		printf("Argument Exception: %s\n",e.what());
	}
	catch (UnauthorizedAccessException& e)
	{
		printf("Unauthorized Access Exception: %s\n",e.what());
	}
	catch (IOException& e)
	{
		printf("IO Exception: %s\n",e.what());
	}
	catch (InvalidOperationException& e)
	{
		printf("Invalid Operation Exception: %s\n",e.what());
	}
	catch (ConfigurationException& e)
	{
		printf("Configuration Exception: %s\n",e.what());
	}
	catch (StreamingException& e)
	{
		printf("Streaming Exception: %s\n",e.what());
	}
	catch (TimeoutException&)
	{
		printf("TimeoutException\n");
	}
	catch (TransportException& e)
	{
		printf("TransportExceptionn: %s\n",e.what());
	}

	catch (NotSupportedException& e)
	{
		printf("NotSupportedException: %s\n",e.what());
	}
}

SoftKineticCamera::SoftKineticCamera() {
	m_isInitialized = Init();
}

SoftKineticCamera::~SoftKineticCamera() {
	cutilSafeCall(cudaFree(m_cudaRGBMap));
	delete[] m_depthMap;
	delete[] m_rgbMap;
	m_context.quit();
}


bool SoftKineticCamera::Init() {
	m_context = Context::create("localhost");
	m_context.deviceAddedEvent().connect(&SoftKineticCamera::onDeviceConnected);
	m_context.deviceRemovedEvent().connect(&SoftKineticCamera::onDeviceDisconnected);

	// Get the list of currently connected devices
	vector<Device> devices = m_context.getDevices();


	if (devices.size()==0)
	{
		return false;
	}

	//Assume that there is only one device
	m_bDeviceFound=true;
	m_device=devices[0];
	m_device.nodeAddedEvent().connect(&SoftKineticCamera::onNodeConnected);
	m_device.nodeRemovedEvent().connect(&SoftKineticCamera::onNodeDisconnected);

	vector<Node> nodes = m_device.getNodes();

	for (int n = 0; n < (int)nodes.size();n++) {
		configureNode(nodes[n]);
	}	

	m_context.startNodes();


	//Allocate memory to depth and color frames.
	int32_t colorWidth,colorHeight;
	int32_t depthWidth,depthHeight;
	FrameFormat_toResolution(m_cnode.getConfiguration().frameFormat,&colorWidth,&colorHeight);
	FrameFormat_toResolution(m_dnode.getConfiguration().frameFormat,&depthWidth,&depthHeight);

	sizeOfDepthMap=depthWidth*depthHeight*sizeof(CAMERA_DEPTH_TYPE);
	m_depthMap=new CAMERA_DEPTH_TYPE[sizeOfDepthMap];
	sizeOfRGBMap=colorWidth*colorHeight*sizeof(CAMERA_RGB_TYPE);
	m_rgbMap=new uint8_t[sizeOfRGBMap];
	this->colorWidth=colorWidth;
	this->colorHeight=colorHeight;

	cutilSafeCall(cudaMalloc((void**)&m_cudaRGBMap, sizeOfRGBMap));
	cutilSafeCall(cudaMalloc((void**)&m_cudaDepthMap, sizeOfDepthMap));


	tthread::thread eventThread(&runEvents,0);
	eventThread.detach();	

	SetParams(SOFTKINETIC_FOH_RADIANS, SOFTKINETIC_FOV_RADIANS,depthWidth,depthHeight);




	m_params.m_z_offset = 0.f;
	m_params.m_min_depth = SOFTKINETIC_MIN_DEPTH_MM;
	m_params.m_max_depth = SOFTKINETIC_MAX_DEPTH_MM;
	m_params.k1 = SOFTKINETIC_K1;
	m_params.k2 = SOFTKINETIC_K2;
	m_params.p1 = SOFTKINETIC_P1;
	m_params.p2 = SOFTKINETIC_P2;
	m_params.depthNullValue = SOFTKINETIC_DEPTH_NULL_VALUE;


	CameraAbstract::Init();

	return true;
}


bool SoftKineticCamera::GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	//TODO: check that the frame is new
	if(d_newDepthMap) {				
		tthread::lock_guard<tthread::mutex> g(depth_mutex);
		cutilSafeCall(cudaMemcpy(m_cudaDepthMap, m_depthMap, sizeOfDepthMap, cudaMemcpyHostToDevice));
		dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
		undistort(m_params,m_cudaDepthMap,d_newDepthMap);
		//cutilSafeCall(cudaMemcpy(d_newDepthMap, m_depthMap, sizeOfDepthMap, cudaMemcpyHostToDevice));

		MedianFilter(d_newDepthMap);

		//GradientFilter(d_newDepthMap);
		AverageFilter(d_newDepthMap);
		return true;
	}
	return false;
}

bool SoftKineticCamera::GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
	////TODO: check that the frame is new
	if (d_newRgbMap) {
		tthread::lock_guard<tthread::mutex> guard(rgb_mutex);
		cutilSafeCall(cudaMemcpy(m_cudaRGBMap, m_rgbMap, sizeOfRGBMap, cudaMemcpyHostToDevice));
		ConvertBGRToRGBA(m_cudaRGBMap,d_newRgbMap);
		return true;
	}
	return false;
}

