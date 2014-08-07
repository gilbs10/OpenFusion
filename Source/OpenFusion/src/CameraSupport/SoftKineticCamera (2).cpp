#include <iostream>	// TODO REMOVE
#include "SoftKineticCamera.h"
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h // TODO REMOVE - NOT MODULAR
#include "CudaUtilities.cuh"

#include <vector>
#include <exception>
#include <Eigen\Core>
#include <Eigen\LU>
#include "DepthSense.hxx"

//TODO: verify
#define SOFTKINETIC_MIN_DEPTH_MM (150.f)
#define SOFTKINETIC_MAX_DEPTH_MM (1000.f)

//Those are the depth fov parameters the color sensor are slightly different
#define SOFTKINETIC_FOV_RADIANS 1.012290966
#define SOFTKINETIC_FOH_RADIANS 1.291543646 


using namespace std;
using namespace DepthSense;
using std::cout;
using std::endl;


Context m_context;
DepthNode m_dnode;
ColorNode m_cnode;
AudioNode m_anode;
Device m_device;

bool m_bDeviceFound = false;




/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    //printf("A#%u: %d\n",m_aFrames,data.audioData.size());
    //m_aFrames++;
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{
    //printf("C#%u: %d\n",g_cFrames,data.colorMap.size());
    //g_cFrames++;
}

/*----------------------------------------------------------------------------*/
// New depth sample event handler
void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    
    //g_dFrames++;

    //// Quit the main loop after 200 depth frames received
    //if (g_dFrames == 200)
    //    g_context.quit();
}

/*----------------------------------------------------------------------------*/
void configureAudioNode()
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
void configureDepthNode()
{
    m_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);

    DepthNode::Configuration config = m_dnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_QVGA;
    config.framerate = 25;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;
    m_dnode.setEnableVertices(true);

    try 
    {
        m_context.requestControl(m_dnode,0);

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
void configureColorNode()
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
void configureNode(Node node)
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
void onNodeConnected(Device device, Device::NodeAddedData data)
{
    configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
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
void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
    if (!m_bDeviceFound)
    {
        data.device.nodeAddedEvent().connect(&onNodeConnected);
        data.device.nodeRemovedEvent().connect(&onNodeDisconnected);
        m_bDeviceFound = true;
    }
}

/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    m_bDeviceFound = false;
    printf("Device disconnected\n");
}

SoftKineticCamera::SoftKineticCamera() {
	m_isInitialized = Init();
}

SoftKineticCamera::~SoftKineticCamera() {
	//todo: you know...
}


bool SoftKineticCamera::Init() {
	m_context = Context::create("localhost");

    m_context.deviceAddedEvent().connect(&onDeviceConnected);
    m_context.deviceRemovedEvent().connect(&onDeviceDisconnected);

    // Get the list of currently connected devices
    vector<Device> devices = m_context.getDevices();

    if (devices.size()==0)
    {
        return false;
	}
    
	m_bDeviceFound=true;
	m_device=devices[0];
	m_device.nodeAddedEvent().connect(&onNodeConnected);
    m_device.nodeRemovedEvent().connect(&onNodeDisconnected);

    vector<Node> nodes = m_device.getNodes();

    for (int n = 0; n < (int)nodes.size();n++) {
        configureNode(nodes[n]);
	}	

    m_context.startNodes();
	m_context.run<<<1,1>>>();

	//TODO: check for depth and color nodes

	int32_t width,height;
	FrameFormat_toResolution(m_dnode.getConfiguration().frameFormat,&width,&height);
		
	SetParams(SOFTKINETIC_FOH_RADIANS, SOFTKINETIC_FOH_RADIANS,width,height);

	m_params.m_z_offset = 0.f;
	m_params.m_min_depth = SOFTKINETIC_MIN_DEPTH_MM;
	m_params.m_max_depth = SOFTKINETIC_MAX_DEPTH_MM;

	//TODO: check what it is doin
	//cutilSafeCall(cudaMalloc((void**)&md_source_rgb, m_params.m_width*m_params.m_height*sizeof(XnRGB24Pixel)));
	
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

bool SoftKineticCamera::GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	/*XnStatus nRetVal;

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
*/
	return true;
}

bool SoftKineticCamera::GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
	/*XnStatus nRetVal;

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
*/
	return true;
}
