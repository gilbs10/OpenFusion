
#ifndef ___CONSTANTS_H_
#define ___CONSTANTS_H_

#include <cfloat> // For FLT_MAX
#include <math.h>

#define MATH_PI (3.14159265358979323846264338327)

#define THREADS_IN_BLOCK (512)
#define CORRESPONDENCE_BLOCK (THREADS_IN_BLOCK)
#define BLOCK_SIZE (16)

#define _EPSILON_ (0.000001f)

// TODO MORE ALGORITHM PARAMS (Camera dependant)
// - world volume
// - MM_FROM_CUBE_FACE
// - Truncation (according to world volume)
// - steps in raycasting (according to world volume)


const unsigned int worldVolume = 1024; // - for primesense. in mm
//const unsigned int worldVolume = 512; // - for gip, in mm.
const unsigned int VOLUME_SIZE = 512;

// TODO !!! IMPORTANT!!  for gip/primesense - change number of steps in raycaster.  gip - 1.0f, primesense - 3.0f

//CAMERAPOSITION
#define MM_FROM_CUBE_FACE (256.f) // - for primesense
//#define MM_FROM_CUBE_FACE (512.f) // - for gip

#define CAMERA_START_Z_MM ((float)worldVolume/2+MM_FROM_CUBE_FACE) //The camera should start 80 cm from the face of the cube.  the cube is worldVolume from face to face, centering around (0,0).
#define CAMERA_START_Z_OPENGL (0.f) // The camera start position z-coord for mouse traveling in the volume

#define BAD_VERTEX (FLT_MAX)

// MAYBE CHANGE THIS TO MAX-MIN
#define VOLUMEWIDTH ((float)worldVolume)

#define VOLUMEMINX (-(float)worldVolume*0.5)
#define VOLUMEMINY (VOLUMEMINX)
#define VOLUMEMINZ (VOLUMEMINX)
#define VOLUMEMAXX ((float)worldVolume*0.5)
#define VOLUMEMAXY (VOLUMEMAXX)
#define VOLUMEMAXZ (VOLUMEMAXX)

#define DISTANCE_THRESHOLD (25.f)
#define NORMAL_THRESHOLD (0.65f)
#define ICP_ITERATIONS (15) // TODO CONSIDER REDUCING THIS NUMBER

#define TRUNCATION (10.f)	// - for primesense. It's about 3cm = 30 mm. // TODO: THIS SHOULD BE A FUNCTION OF THE VOLUME SIZE
//#define TRUNCATION (10.f)	// - for gip
#define MAX_WEIGHT (120)		// WE THINK THIS MEANS HOW MANY FRAMES UNTIL AN OBJECT DISAPPERAS. 60 IS ABOUT TWO SECONDS IDEALLY
#define MIN_WEIGHT (0)			// The minimum weight possible for TSDF



// The parameters of the Asus: min and max reliable depth values in mm
//#define CAMERA_DEPTH_MIN_MM (300.f) // - for primesense

//#define CAMERA_DEPTH_MIN_MM (10.f) // - for softkinetic
//#define CAMERA_DEPTH_MAX_MM (6000.f) // - for primesense

// GIP // CENTER IS ABOUT 739.1
//#define CAMERA_DEPTH_MIN_MM (440.f) // - for gip
//#define CAMERA_DEPTH_MAX_MM (840.f) // - for gip


////// PROJECTION MATRIX

const float focalLengthX = (float)(tan(29.0*MATH_PI/180)*2);			// horizontal focal length. ASUS XTIONPRO LIVE: 58 degrees.
const float focalLengthY = (float)(2*sqrt(2.0)-2);					// vertical focal length. ASUS XTIONPRO LIVE: 45 degrees.

// The computations were taken from "xnConvertProjectiveToRealWorld" of xnOpenNi.cpp
// We took the final calculations and put the coefficients to matrix form
// REALLY IMPORTANT: You should multiply by (x,y,1) coordinates of the camera space, where x = x_camera/z_camera, y = y_camera/z_camera
//					 The inverse matrix will return the same (x,y) given a pixel, so you have to multiply by z_camera to get the correct (x,y) of the camera space.
//					 Same instructions to the inverse! ( z in the multiplying vector is always 1 )

// We didn't use K=(fx,  0,  cx)
//                 (0,  fy,  cy)
//                 (0,   0,   1)
// as specified in sites we found on the internet because when multiplying (x,y,z) the value of z is taken into consideration when it shouldn't be


// K matrix (projection matrix)              	// TODO!!! maybe the picture will be upside down. Just need to put a (-) before the y-coefficient



/*const float KMat[9] = {-IMAGE_W/focalLengthX,                0,                  IMAGE_W/2.0,
						 0,                   -IMAGE_H/focalLengthY,				 IMAGE_H/2.0,
						 0,                                   0,                          1.0};

//Inverse K matrix (inverse projection matrix)
const float KinvMat[9] = {-focalLengthX/IMAGE_W,             0,				focalLengthX*0.5,
						   0,					-focalLengthY/IMAGE_H,		focalLengthY*0.5,
						   0,								 0,							  1.0};
*/


/////////////// ASUS


/*const float KMat[9] = {-570.342163,                0,                  320.f,
						 0,                   -570.342224,				 240.f,
						 0,                                   0,                          1.0};

//Inverse K matrix (inverse projection matrix)
const float KinvMat[9] = {-0.001753333463442,             0,			0.561066708301557,
						   0,					-0.001753333275918,		0.420799986220203,
						   0,								 0,			  1.0};
*/					   

/////////////////// KINECT
/*
const float KMat[9] = {-525.f,                0,                  320.f,
						 0,                   -525.f,			  240.f,
						 0,                   0,                          1.0};

//Inverse K matrix (inverse projection matrix)
const float KinvMat[9] = {-0.0019047619,             0,			        0.60952380952381,
						   0,					-0.001904761904762,		0.457142857142857,
						   0,								 0,			  1.0};
*/

#endif //___CONSTANTS_H_
