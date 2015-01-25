#ifndef ___CAMERAABSTRACT_H_
#define ___CAMERAABSTRACT_H_

#include "utilities.h"


/**
	\brief Generic camera interface

	This class represent a camera connected to the system.
	The class provide an interface to get the depth/RGB frames from the camera.
*/

class CameraAbstract {

protected:
	bool m_isInitialized;
	CameraParams m_params;
	/**
		\brief Used by the filter function too store the original frame being filterd.
		\sa AverageFilter(), GradientFilter(), MedianFilter()
	*/
	CAMERA_DEPTH_TYPE* m_tempDepthArray;
	/**
		\brief Used by the filter function too store the original frame being filterd.
		\sa AverageFilter(), GradientFilter(), MedianFilter()
	*/
	CAMERA_RGB_TYPE* m_tempRgbArray;
	
	/**
		\brief Calculates and sets camera's params. Not virtual.
	*/
	void SetParams(float foh, float fov, uint width, uint height);
	/** 
	\brief Initializes camera, creates parameters struct.

		Very Important!!!! CameraAbstract::Init() must be called from the end of each child class's Init() implementation.
		It uses the cameraParams width and height which are only known at the end of the child's Init().
	*/
	virtual bool Init();

	/** \brief Fix distortion coused by the lens.

		based on Aaron solution described <a href="http://stackoverflow.com/questions/12117825/how-can-i-undistort-an-image-in-matlab-using-the-known-camera-parameters">here.</a>
	*/
	void undistort(CameraParams m_params, 
				   CAMERA_DEPTH_TYPE* depthMap,  CAMERA_DEPTH_TYPE* newDepthMap);
	
	/** \brief An average filter for a depth frame */
	void AverageFilter(CAMERA_DEPTH_TYPE* d_newDepthMap);
	/** \brief A gradient filter for a depth frame */
	void GradientFilter(CAMERA_DEPTH_TYPE* d_newDepthMap);
	/** \brief An Median filter for a depth frame */
	void MedianFilter(CAMERA_DEPTH_TYPE* d_newDepthMap);

public:

	/** \brief The size of a depth frame in bytes */
	int m_depthSizeBytes;
	/** \brief The size of a color frame in bytes */	
	int m_rgbSizeBytes;
	
	/** \brief Post: camera parameters are initialized. */
	CameraAbstract();
	
	// Virtual destructor.
	virtual ~CameraAbstract();
	
	/** \brief Getter.*/
	bool IsInitialized();
	
	/** \brief Getter.*/
	CameraParams* GetParameters();

	/** \brief Get new depth frame. */
	virtual bool GetDepthFrame(CAMERA_DEPTH_TYPE* d_newDepthMap) = 0;

	virtual char* GetCameraTypeStr() = 0;

	/** \brief Get new rgb frame, if available. */
	virtual bool GetRgbFrame(CAMERA_RGB_TYPE* d_newRgbMap) {
		return false;
	};

	// TODO: Add functions readToFile, writeToFile in the abstract class. 
	// notice that we need to add a configurations file for each camera when it writes to file.

};

#endif // ___CAMERAABSTRACT_H_

