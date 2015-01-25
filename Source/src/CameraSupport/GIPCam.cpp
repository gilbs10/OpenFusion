#include <iostream>	// TODO REMOVE

#include "GIPCam.h"
#include "GIPCam\gipFile.h"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>

#include <Eigen\Core>
#include <Eigen\LU>

#define DATA_FILE_PATH "C:\\Development\\NVIDIA GPU Computing SDK 4.0\\C\\src\\volumeRender\\GIPCam - Data\\aaron1.v3r" // TODO: MAKE THIS PATH PLATFORM COMPATIBLE (using filesystemhelper.h). Maybe get from the user.
//#define DATA_FILE_PATH "S:\\3DCAM\\Dewalt-Alex\\DeWALT-Alex.v3r" // Today the S: drive wasn't connected.
#define CALIBRATION_FILE_PATH "C:\\Development\\NVIDIA GPU Computing SDK 4.0\\C\\src\\volumeRender\\GIPCam - Data\\calibration_CB_Ex.log"

// Image coordinates for GIPCam (normalized pixels).
#define MIN_X (-1.f)
#define MAX_X ( 1.f)
#define MIN_Y (-(4.f)/(3.f))
#define MAX_Y ( (4.f)/(3.f))

#define GIP_MIN_DEPTH_MM (440.f);
#define GIP_MAX_DEPTH_MM (840.f);
#define GIP_Z_OFFSET_MM (705.4f);

using std::cout; // TODO REMOVE
using std::endl;

GIPCam::GIPCam() : m_currentFrameNumber(0), m_zCoordsMatrixSize(0) {
	m_isInitialized = Init();

}

GIPCam::~GIPCam() {
	cutilSafeCall(cudaFree(m_zCoordsMatrix));
}


bool GIPCam::Init() {
	SCANNER_RESULT res;

	// Get parameters of image.
	CalibrationParameters calibration;
	ImageData_GIP data;
	res = readFileGIP(DATA_FILE_PATH, &data, 0, 0);	// Read first frame.
	if (res != SCANNER_OK) {
		clearImageDataGIP(&data);
		cout << "Can't read GIP image parameters." << endl;
		return false;
	}
	
	//res = readCalibrationParametersFromGipFile(CALIBRATION_FILE_PATH, &calibration, &data); // Read calibration.
	//if (res != SCANNER_OK) {
	//	clearImageDataGIP(&data);
	//	return false;
	//}

	const int width = 480; 
	const int height = 640;
	
	// TODO: WE GUESS WE USE THE CANONICAL CALIBRATION, WHICH MEANS K IS MULTIPLIED BY [R|t] WHERE R IS IDENTITY AND t IS ZEROS.

	// TODO!!!!!!!!!!!!!!!!! READING FROM FILE WASN'T SUCCESSFULL. WE DECIDED TO USE THE HARD-CODED VALUES.

	const float fx = 7.174f; //calibration.K1[0]; // Extract K matrix
	const float fy = 7.124f; //calibration.K1[4];
	const float cx = 0.099f; //0.058; //calibration.K1[2];
	const float cy = -0.049f; //0.148; //calibration.K1[5];
	
	SetParams(fx, fy, cx, cy, width, height);

	m_params.m_height = data.header.cols; // cols=480
	m_params.m_width = data.header.rows; // rows=360. TODO THAT'S WEIRD I KNOW

	m_params.m_z_offset  = GIP_Z_OFFSET_MM;
	m_params.m_min_depth = GIP_MIN_DEPTH_MM;
	m_params.m_max_depth = GIP_MAX_DEPTH_MM;

	clearImageDataGIP(&data);

	m_zCoordsMatrixSize = m_params.m_width*m_params.m_height*sizeof(float);
	cutilSafeCall(cudaMalloc((void**)&m_zCoordsMatrix, m_zCoordsMatrixSize));

	CameraAbstract::Init();

	return true;
}

/////// TODO. VERED: I RECALCULATED THIS MATRIX. MY CONCLUSION IS -
//
//	[1  0  180][240  0  0][U]			[1  0  180][240  0  0]     [X]
//  [0  1  240][ 0 -320 0][V]     = 	[0  1  240][ 0 -320 0]x C x[Y]
//  [0  0   1 ][ 0   0  1][W]			[0  0   1 ][ 0   0  1]     [Z]
//																   [W]
//  \_______________________/			\________________________/
//		THIS IS REAL PIXEL				   THIS IS THE NEW MATRIX (NOTE THAT CX,CY MUST BE ZERO IN THE ORIGINAL C. OTHERWISE WEIRD THINGS HAPPEN.
//		([U V W] IS NORMALIZED)
//											THEN WE NEED TO CONVERT [X,Y,Z,W] INTO MM INSTEAD OF DECIMETERS. THIS IS DONE ONCE.
//


void GIPCam::SetParams(float fx, float fy, float cx, float cy, uint width, uint height) {
	// TODO: NOTE: WE IGNORE THE SKEW TERM IN THE GIVEN K MATRIX

	const float scale_x = 240.f; //      // width*( 1.f/(MAX_X-MIN_X) ); // width*0.5.
	const float scale_y = 240.f; // height*( 1.f/(MAX_Y-MIN_Y) ); // height*0.75*0.5.
	
	const float new_fx = scale_x * fx;
	const float new_fy = scale_y * fy;

	const int new_cx = (int)(scale_x*cx + 180.f); // (float)width/2;
	const int new_cy = (int)(scale_y*cy + 240.f); //(float)height/2;

	//m_params.m_height = height;
	//m_params.m_width = width;
	m_params.m_intrinsic.Set(-new_fx, -new_fy, new_cx, new_cy);

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> intrinsic;
	intrinsic.setZero();
	intrinsic(0,0) = m_params.m_intrinsic.m_fx;
	intrinsic(1,1) = m_params.m_intrinsic.m_fy;
	intrinsic(2,2) = 1.f;
	intrinsic(0,2) = m_params.m_intrinsic.m_cx;
	intrinsic(1,2) = m_params.m_intrinsic.m_cy;

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> invIntrinsic = intrinsic.inverse();

	m_params.m_invIntrinsic.Set(invIntrinsic(0,0), invIntrinsic(1,1), invIntrinsic(0,2), invIntrinsic(1,2));
}

////////////////////////////////////////////// TODO //////////////////////////////////////

//MIN_DEPTH SHOULD BE A PARAMETER OF THE CAMERA IN ADDITION TO THE WIDTH,HEIGHT AND K!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////////////////////

bool GIPCam::GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	if (m_currentFrameNumber > 142) { // TODO IF FROM FILE, READ THE HEADER OF THE DATA AND FIND THE TOTAL FRAMES
		// aaron1: 142
		// alon1: 188

		return false;
	}

	SCANNER_RESULT res;

	ImageData_GIP data;
	res = readFileGIP(DATA_FILE_PATH, &data, m_currentFrameNumber, 0);
	if (res != SCANNER_OK) {
		clearImageDataGIP(&data);
		return false;
	}

	/*if (currentFrameNumber == 50) {
		ofstream outfile("gipimage.txt", ofstream::binary);
		outfile.clear();

		// write to outfile
		outfile.write ((const char*)data.z, m_params.m_width*m_params.m_height*sizeof(float));
		outfile.flush();
		outfile.close();
	}*/


	if (d_newDepthMap) {

		cutilSafeCall(cudaMemcpy(m_zCoordsMatrix, data.z, m_zCoordsMatrixSize, cudaMemcpyHostToDevice));

		ConvertDepthMap(m_zCoordsMatrix, d_newDepthMap, m_params.m_width, m_params.m_height, m_params.m_z_offset); // TODO - CHECK THAT WIDTH=360 AND HEIGHT=480.
		
		cutilSafeCall(cudaDeviceSynchronize());
		 // TWO ITERATIONS OF NOISE REDUCTION. MAYBE PASS THE THRESHOLD AS PARAMETER AND CHANGE IT.
		MedianFilter(d_newDepthMap);

		GradientFilter(d_newDepthMap);
		AverageFilter(d_newDepthMap);
		AverageFilter(d_newDepthMap);
		//TODO - ask Aaron about filters


		//cutilSafeCall(cudaMemcpy(d_newDepthMap, data.z, IMAGE_W*IMAGE_H*sizeof(CAMERA_DEPTH_TYPE), cudaMemcpyHostToDevice)); 
	}

	clearImageDataGIP(&data);
	m_currentFrameNumber++; // CHECK IF THIS FRAME EXISTS // TODO SAME FRAME ALL THE TIME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	return true;
}


