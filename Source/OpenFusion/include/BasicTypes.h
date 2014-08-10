
#ifndef ___BASIC_TYPES
#define ___BASIC_TYPES

#include "Constants.h"
// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

//OpenGL Graphics includes
#include "GL/glew.h"
#if defined (__APPLE__) || defined(MACOSX)
#include "GLUT/glut.h"
#else
#include "GL/freeglut.h"
#endif
////////////////////////////This was for pbo data struct////////////////////////////////////


typedef unsigned int uint;
typedef unsigned char uchar;


typedef unsigned short CAMERA_DEPTH_TYPE;
typedef uint CAMERA_RGB_TYPE;

static const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

// Voxels Size and type declaration
typedef struct {
	/**
	\brief Stands for Truncated Surface Distance Function.
	*/
	float tsdf;
	//char weight;
} Voxel;

typedef Voxel VoxelType;
typedef unsigned char VoxelWeightType;


typedef struct {
	float3* vertex;
	float3* normal;
} PointInfo;

typedef struct {
	CAMERA_DEPTH_TYPE* depth;
	float3* vertex;
	float3* normal;
} CameraData;



////////////////PUT THIS IN VOLUMERENDER.CPP
typedef struct _PboData
{
	GLuint pbo;  // OpenGL pixel buffer object
	GLuint tex;  // OpenGL texture object
	struct cudaGraphicsResource *cuda_pbo_resource;  // CUDA Graphics Resource (to transfer PBO)

	_PboData() : pbo(0), tex(0), cuda_pbo_resource(NULL)
	{
		//empty
	}
} PboData;

////////////////////////////////////////////////////////////////////////////////////////////



// intrinsic calibration matrix
typedef struct Intrinsic_t {
	float m_fx;
	float m_fy;
	float m_cx;
	float m_cy;

	void Set(float fx = 0, float fy = 0, float cx = 0, float cy = 0) {
		m_fx = fx;
		m_fy = fy;
		m_cx = cx;
		m_cy = cy;
	}
} Intrinsic;

// camera parameters
typedef struct CameraParams_t {
	Intrinsic m_intrinsic;
	Intrinsic m_invIntrinsic;
	uint  m_width;
	uint  m_height;
	float m_z_offset;
	float m_min_depth; // in mm
	float m_max_depth; // in mm
	float k1;
	float k2;
	float k3;
	float p1;
	float p2;	
	CAMERA_DEPTH_TYPE depthNullValue;
} CameraParams;



#endif //___BASIC_TYPES