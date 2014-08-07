#include "cutil_inline.h"    // includes cuda.h and cuda_runtime_api.h
#include "FileSystemHelper.h"
#include "utilities.h"
#include "CameraHandler.h"
#include "PrimesenseCamera.h"
#include "GIPCam.h"
#include "SoftKineticCamera.h" // TODO - remove comment when softkinetic module is returned
#include "CameraFromFile.h"
#include "WriteToFile.h"


#include <fstream> // TODO REMOVE
#include <iostream>

#include <stdlib.h> // TODO IS NEEDED?

#include <time.h> // TODO REMOVE

#define NUM_FRAMES (10000)

//using namespace cv;
using namespace std;

extern "C" void updateCameraData(dim3 gridSize,
								 dim3 blockSize,
								 CameraData* d_cameraData, 
								 CAMERA_DEPTH_TYPE* d_new_depth,
								 float* d_currentTransform, 
								 uint* newNormalMap);

extern "C" void copyKInvMatrixCamera(const float *_kInvMat);

void help()
{ // TODO: UPDATE HELP. IT'S NO LONGER CV TYPES.
        cout << "\nThis program demonstrates usage of depth sensors (Kinect, XtionPRO,...).\n"
                        "The user gets some of the supported output images.\n"
            "\nAll supported output map types:\n"
            "1.) Data given from depth generator\n"
            "   OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
            "   OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
            "   OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
            "   OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
            "   OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
            "2.) Data given from RGB image generator\n"
            "   OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
            "   OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
         << endl;
}


void CameraHandler::readInputFromCamera(CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_RGB_TYPE* d_newRgbMap)
{
	// TODO: Extract each one of them to a seperate function

	bool res = m_camera->GetDepthFrame(d_newDepthMap);
	if (!res) {
		cout << "Depth retrieval failed!" << endl;
	}
		// TODO CALL CUDA AND STUFF

	res = m_camera->GetRgbFrame(d_newRgbMap);
	if (!res) {
		cout << "RGB retrieval failed! Remove logging." << endl;
	}


	return;
}

bool isFromCamera = true;
void CameraHandler::cameraIteration(CAMERA_DEPTH_TYPE* d_newDepthMap,
									CAMERA_RGB_TYPE* newRgbMap,
									uint* newDepthMap,
									const CameraData& d_cameraData,
									float* d_currentTransform,
									uint* newNormalMap,
									dim3 gridSize,
									dim3 blockSize) {

	// IMPORTANT: Only one from the lines below should be uncommented!
	// The read functions return one frame, while the write function writes all frames at once.


	if(!isFromCamera)
//		readInputFromFile(d_newDepthMap, newRgbMap);
		cout << "READ INPUT FROM FILE IS DEPRECATED! USE THE CLASS 'CAMERA FROM FILE'." << endl;
	else{
		readInputFromCamera(d_newDepthMap, newRgbMap);
		//writeInputToFile();
	}
	updateCameraData(gridSize, blockSize, d_cameraData, d_newDepthMap, d_currentTransform, newNormalMap, newDepthMap);
}

CameraHandler::CameraHandler()
{
	inputData_depth = new CAMERA_DEPTH_TYPE[m_params.m_width*m_params.m_height];
	inputData_rgb = new CAMERA_RGB_TYPE[m_params.m_width*m_params.m_height];
	
	if(isFromCamera) {
		char path[200] = {0};
		FileHandler* fileHandler = FileHandler::Init();
		fileHandler->GetPath("Video", "realTrial", path); // TODO: What is this real trial? You may want to get it from the user.

		// TODO: THE SOFTKINETIC CONSTRUCTOR MUST CHECK THAT IT CAN READ FROM THE SERVER.
		// OTHERWISE IT SHOULD RETURN FALSE.

		if ((m_camera = new PrimesenseCamera()) && m_camera->IsInitialized() ||
	//		(m_camera = new GIPCam()) && m_camera->IsInitialized() ||
			(m_camera = new SoftKineticCamera()) && m_camera->IsInitialized() ||	
			(m_camera = new CameraFromFile(path)) && m_camera->IsInitialized())
		{
			m_params = *(m_camera->GetParameters());
		}
		else {
			cout << "No camera found!" << endl;
			exit(1); // TODO CHANGE
		}
	}
}


CameraHandler::~CameraHandler () {

	delete m_camera;

	delete[] inputData_depth;
	delete[] inputData_rgb;
}

void CameraHandler::SwitchToWriteMode() {
	CameraAbstract* old_camera = m_camera;
	m_camera = new WriteToFile(old_camera);
	if (!m_camera->IsInitialized()) {
		cout << "Camera write initialization failed!" << endl;
		exit(1); //TODO CHANGE
	}
}

void CameraHandler::SwitchToReadMode() {
	delete m_camera;
	cout << "Please enter the directory name of your recorded video: ";
	char dirName[150];
	cin >> dirName;
	FileHandler* fileHandler = FileHandler::Init();
	const char* videoSuperDirectoryName = "Video";
	char fullPath[150];
	fileHandler->GetPath(videoSuperDirectoryName, dirName, fullPath);

	m_camera = new CameraFromFile(fullPath);
	if (!m_camera->IsInitialized()) {
		cout << "Failed initializing read from file." << endl;
		delete fileHandler;
		exit(1);
	}
	delete fileHandler;
}

// TODO:
// 2. SINGLE INTERFACE TO CAMERAHANDLER: READ FROM CAMERA, READ FROM FILE, WRITE TO FILE ; CAMERA TYPE (LATER- MAKE AUTOMATIC).
// 3. EXTRACT PARAMETERS FROM THE CAMERA MODULE, AND SUBSTITUTE IT TO THE REST OF THE PROJECT (IMAGE SIZE + INTRINSIC). - TODAY 3/4/2013!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// 4. CREATE SHARED MEMORY AND USE BOOST (LINK BOOST) FOR EVERY CAMERA MODULE.
// 5. IMPLEMENT OTHER CAMERA MODULES (CHECK FOR KINECT! - V).