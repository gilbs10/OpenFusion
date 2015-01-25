#include "WriteToFile.h"

#include <iostream>
#include <ctime>
#include "cutil_inline.h"    // includes cuda.h and cuda_runtime_api.h
#include "FileSystemHelper.h"

using std::cout;
using std::cin;
using std::endl;

WriteToFile::WriteToFile(CameraAbstract* camera) : m_camera(camera), m_frameNumber(1) {
	m_isInitialized = Init();
}

WriteToFile::~WriteToFile() {

	cout << "Writing configuration file...";

	m_settingsFile.clear();
	CameraParams* params = m_camera->GetParameters();
	
	//Print the following to the settings file:
	//Image width (pixels)
	//Image height (pixels)
	//Intrinsic-fx Intrinsic-fy
	//Intrinsic-cx Intrinsic-cy
	//Z offset in mm (for GIP, otherwise 0)
	//Camera min depth in positive mm
	//Camera max depth in positive mm
	//Number of frames
	m_settingsFile << params->m_width << endl;
	m_settingsFile << params->m_height << endl;
	m_settingsFile << params->m_intrinsic.m_fx << " " << params->m_intrinsic.m_fy << endl;
	m_settingsFile << params->m_intrinsic.m_cx << " " << params->m_intrinsic.m_cy << endl;
	m_settingsFile << params->m_z_offset << endl;
	m_settingsFile << params->m_min_depth << endl;
	m_settingsFile << params->m_max_depth << endl;
	m_settingsFile << m_frameNumber - 1 << endl;
	
	//Print the following to the settings file:
	//Name
	//Date taken
	//Camera type
	m_settingsFile << m_dirName << endl;
	
	// get and print system date and time
    time_t rawtime;
    struct tm * timeinfo;
    char timeBuffer [80];

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    strftime (timeBuffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);
	m_settingsFile << timeBuffer << endl;
	m_settingsFile << m_camera->GetCameraTypeStr() << endl;
	
	m_settingsFile.flush();
	m_settingsFile.close();
	
	cout << "done" << endl;

	delete m_camera;
	delete[] m_inputDataDepth;
}

bool WriteToFile::Init() {
	// Creating new directory for the video.
	cout << "Please enter a name for your video directory (up to 100 chars): ";
	char userDirName[100]; // TODO MAKE ALL THOSE NUMBERS (APPEAR ALL OVER THE PROJECT, FILE-RELATED) TO DEFINES
	cin >> userDirName;
	
	const char* videoSuperDirectoryName = "Video"; // TODO OPTIONA: MAKE THIS "VIDEO" CONST STRING TO CLASS MEMBER. IT APPEARS ALSO IN THE READ FUNCTION.
	FileHandler* fileHandler = FileHandler::Init();
	if (!fileHandler->CreateDir(videoSuperDirectoryName)) {
		cout << "Failed to create super directory.";
		delete fileHandler;
		return false;
	}

	fileHandler->GetPath(videoSuperDirectoryName, userDirName, m_dirName);

	if (!fileHandler->CreateDir(m_dirName)) {
		cout << "Failed to create directory." << endl;
		delete fileHandler;
		return false;
	}

	// Creating settings file.
	char settingsFileName[200];
	fileHandler->GetPath(m_dirName, "Parameters.txt", settingsFileName);
	
	m_settingsFile.open(settingsFileName);
	if(!m_settingsFile.is_open()) {
		cout << "Failed to open file [" << settingsFileName << "]" << endl;
		delete fileHandler;
		return false;
	}

	m_inputDataDepth = new CAMERA_DEPTH_TYPE[m_camera->GetParameters()->m_width * m_camera->GetParameters()->m_height];
	m_inputDataRgb = new CAMERA_RGB_TYPE[m_camera->GetParameters()->m_width * m_camera->GetParameters()->m_height];

	//CameraAbstract::Init(); // TODO we dont' call this because we don't need writeToFile to have a cuda pointer.
								// we assume that we won't need to access the unallocated pointer...

	m_depthSizeBytes = m_camera->m_depthSizeBytes;
	m_rgbSizeBytes = m_camera->m_rgbSizeBytes;
	delete fileHandler;
	return true;
}

CameraParams* WriteToFile::GetParameters() {
	return m_camera->GetParameters();
}

bool WriteToFile::GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	if(!m_camera->GetDepthFrame(d_newDepthMap)) {
		return false;
	}

	if (!d_newDepthMap) {
		return true;
	}

	// Construt depth file path.
	char fileName[20] = {0};
	sprintf(fileName, "recorded%d", m_frameNumber);
	char fullPath[250] = {0};
	FileHandler* fileHandler = FileHandler::Init();
	fileHandler->GetPath(m_dirName, fileName, fullPath);

	m_frameNumber++;

	ofstream outFile(fullPath, ofstream::binary);
	outFile.clear();
	cutilSafeCall(cudaMemcpy(m_inputDataDepth, d_newDepthMap, m_depthSizeBytes, cudaMemcpyDeviceToHost));

	// Write to outfile.
	outFile.write ((const char*)m_inputDataDepth, m_depthSizeBytes);

	outFile.flush();
	outFile.close();
	return true;
}

bool WriteToFile::GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
	if(!m_camera->GetRgbFrame(d_newRgbMap)) {
		return false;
	}

	if (!d_newRgbMap) {
		return true;
	}

	// Construt depth file path.
	char fileName[20] = {0};
	// Note that currentFrameNumber is incremented in GetDepthFrame,
	// so we use the previous frame number here.
	sprintf(fileName, "recorded-rgb%d", m_frameNumber - 1);
	char fullPath[250] = {0};
	FileHandler* fileHandler = FileHandler::Init();
	fileHandler->GetPath(m_dirName, fileName, fullPath);

	ofstream outFile(fullPath, ofstream::binary);
	outFile.clear();
	cutilSafeCall(cudaMemcpy(m_inputDataRgb, d_newRgbMap, m_rgbSizeBytes, cudaMemcpyDeviceToHost));

	// Write to outfile.
	outFile.write ((const char*)m_inputDataRgb, m_rgbSizeBytes);

	outFile.flush();
	outFile.close();
	return true;
}
