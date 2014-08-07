#include <iostream>	// TODO REMOVE
#include <fstream> // TODO DON'T REMOVE
#include "CameraFromFile.h"
#include "FileSystemHelper.h"
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h // TODO REMOVE - NOT MODULAR
#include <Eigen\Core>
#include <Eigen\LU>

using std::ifstream;
using std::cout;
using std::endl;

CameraFromFile::CameraFromFile(char* path) :  m_currentFrameNumber(1) {
	sprintf(m_path, "%s", path);
	m_isInitialized = Init();	
}

CameraFromFile::~CameraFromFile() {
	delete[] m_inputDataDepth;
}

bool CameraFromFile::Init() {
	
	char settingsFileName[300];
	FileHandler* fileHandler = FileHandler::Init();
	fileHandler->GetPath(m_path, "Parameters.txt", settingsFileName);
	cout <<"settings filename = " << settingsFileName<<endl;
	ifstream settingsFile(settingsFileName);
	if(!settingsFile.is_open()) {
		cout << "Failed to open file [" << settingsFileName << "]" << endl;
		return false;
	}

	settingsFile >> m_params.m_width >> m_params.m_height;
	settingsFile >> m_params.m_intrinsic.m_fx >> m_params.m_intrinsic.m_fy >> m_params.m_intrinsic.m_cx >> m_params.m_intrinsic.m_cy;
	settingsFile >> m_params.m_z_offset;
	settingsFile >> m_params.m_min_depth;
	settingsFile >> m_params.m_max_depth;

	cout << "width: "<< m_params.m_width << " height: " << m_params.m_height << "depth: " << m_params.m_max_depth<< endl;

	// Inverse intrinsic
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> intrinsic;
	intrinsic.setZero();
	intrinsic(0,0) = m_params.m_intrinsic.m_fx;
	intrinsic(1,1) = m_params.m_intrinsic.m_fy;
	intrinsic(2,2) = 1.f;
	intrinsic(0,2) = m_params.m_intrinsic.m_cx;
	intrinsic(1,2) = m_params.m_intrinsic.m_cy;
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> invIntrinsic = intrinsic.inverse();
	m_params.m_invIntrinsic.Set(invIntrinsic(0,0), invIntrinsic(1,1), invIntrinsic(0,2), invIntrinsic(1,2));

	settingsFile.close();

	// TODO DEBUG
	//cout << "Parameters" << endl;
	//cout << "width = " << m_params.m_width << ", height = " << m_params.m_height << endl;
	//cout << m_params.m_intrinsic.m_fx << " " << m_params.m_intrinsic.m_fy << " " << m_params.m_intrinsic.m_cx << " " << m_params.m_intrinsic.m_cy << endl;

	m_inputDataDepth = new CAMERA_DEPTH_TYPE[m_params.m_width*m_params.m_height];
	m_inputDataRgb = new CAMERA_RGB_TYPE[m_params.m_width*m_params.m_height];

	CameraAbstract::Init();

	return true;
}

bool CameraFromFile::GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	// Construt depth file path.
	char fileName[20] = {0};
	sprintf(fileName, "recorded%d", m_currentFrameNumber);
	char fileName_depth[250] = {0};
	FileHandler* fileHandler = FileHandler::Init();
	fileHandler->GetPath(m_path, fileName, fileName_depth);

	ifstream inputFileDepth(fileName_depth, ifstream::binary);
	if(!inputFileDepth.is_open()) {
		cout << "Failed to open file [" << fileName_depth << "]" << endl;
		return false;
	}

	inputFileDepth.read(reinterpret_cast<char*>(m_inputDataDepth), m_depthSizeBytes);
	inputFileDepth.close();

	if (d_newDepthMap) {
		cutilSafeCall(cudaMemcpy(d_newDepthMap, m_inputDataDepth, m_depthSizeBytes, cudaMemcpyHostToDevice)); 
	}

	m_currentFrameNumber++;

	return true;
}

bool CameraFromFile::GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
	// Construt depth file path.
	char fileName[20] = {0};
	// Note that currentFrameNumber is incremented in GetDepthFrame,
	// so we use the previous frame number here.
	sprintf(fileName, "recorded-rgb%d", m_currentFrameNumber - 1);
	char fileName_rgb[250] = {0};
	FileHandler* fileHandler = FileHandler::Init();
	fileHandler->GetPath(m_path, fileName, fileName_rgb);

	ifstream inputFileRgb(fileName_rgb, ifstream::binary);
	if(!inputFileRgb.is_open()) {
		cout << "Failed to open file [" << fileName_rgb << "]" << endl;
		return false;
	}

	inputFileRgb.read(reinterpret_cast<char*>(m_inputDataRgb), m_rgbSizeBytes);
	inputFileRgb.close();

	if (d_newRgbMap) {
		cutilSafeCall(cudaMemcpy(d_newRgbMap, m_inputDataRgb, m_rgbSizeBytes, cudaMemcpyHostToDevice)); 
	}

	return true;

}