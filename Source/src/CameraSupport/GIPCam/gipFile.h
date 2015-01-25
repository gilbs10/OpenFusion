/****************************************************
* gipFile.h											*
*                                                   *
* (C) GIP Laboratory								*
*     Computer Science Faculty, Technion.	2013	*
****************************************************/

#ifndef _GIP_FILE_H 
#define _GIP_FILE_H 

#define PROJECT_EXPORTS

#if defined(_WIN32) || defined(_WIN64)
	// Windows
	#ifdef PROJECT_EXPORTS
		#define PROJECT_API __declspec( dllexport )
	#else
		#define PROJECT_API __declspec( dllimport )
	#endif

	#include <windows.h>
	#include <io.h>

#else
	// Linux
	#define PROJECT_API __attribute__ ((visibility("default")))
#endif


#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
//#include <sys/types.h>
#include <sys/stat.h>


#define DEFAULT_IS_FILTERED (false)
#define DEFAULT_FRAME_NUMBER (0)

#define DEFAULT_ROWS (480)
#define DEFAULT_COLS (640)
#define DEFAULT_ROWS_TEXTURE (480)
#define DEFAULT_COLS_TEXTURE (640)
#define DEFAULT_BYTES_PER_PIXEL (1)
#define DEFAULT_PIXEL_FORMAT (1)
#define DEFAULT_NUM_TEXTURES (1)
#define DEFAULT_TEXTURE_SEQUENCE (0)

// !!! note change DEFAULT_HEADER_SIZE if any changes in struct ImageData_GIPHeader
// currently DEFAULT_HEADER_SIZE is 125 bytes but sizeof(ImageData_GIPHeader) returns 128 !!!
#define DEFAULT_HEADER_SIZE (125)

#define HEADER_OFFSET_SHIFT (65536)		// Used to separate 16 MSB / 16 LSB of 4 byte int field.
#define CORRUPTED_FILE_SIZE (8601613)	// size of corrupted files - these files have 12 textures but header states only one texture


/***************************************************************************************
*	typedef enum SCANNER_RESULT
*	
* Description
*	result enumeration
* Remarks
*
***************************************************************************************/
typedef enum { 
	SCANNER_OK,
	SCANNER_FAIL,
	SCANNER_INVALID_ARGUMENTS,
	SCANNER_MEMORY_ALLOCATION_FAILURE,

	SCANNER_FILE_OPEN_FAILURE,
	SCANNER_FILE_READ_FAILURE,
	SCANNER_FILE_WRITE_FAILURE,
	SCANNER_FILE_NOT_VALID,	

	SCANNER_ALREADY_INITIALIZED,

	SCANNER_CALIBRATION_FILE_NOT_FOUND,	

	SCANNER_CAMERA_BUS_ENUMERATION_FAILURE,
	SCANNER_CAMERA_INIT_FAILURE,
	SCANNER_CAMERA_GRABBING_FAILURE,
	SCANNER_CAMERA_START_GRABBING_FAILURE,
	SCANNER_CAMERA_STOP_GRABBING_FAILURE,
	SCANNER_CAMERA_BUFFER_OVERFLOW,
	SCANNER_CAMERA_NOT_INITIALIZED,
	SCANNER_CAMERA_TRIGGER_FAILURE,
	SCANNER_CAMERA_GET_PROPERTY_FAILURE,
	SCANNER_CAMERA_SET_PROPERTY_FAILURE,
	SCANNER_CAMERA_SET_REGISTER_FAILURE,
	SCANNER_CAMERAS_SYNCHRONIZATION_FAILURE,

	SCANNER_PROJECTOR_NOT_INITIALIZED,
	SCANNER_PROJECTOR_ALREADY_INITIALIZED,
	SCANNER_PROJECTOR_ALREADY_ACTIVE,
	SCANNER_PROJECTOR_WINDOW_FAILURE,
	SCANNER_PROJECTOR_DEVICE_FAILURE,
	SCANNER_PROJECTOR_DX_FAILURE,
	SCANNER_PROJECTOR_FULL_MODE_FAILURE,
	SCANNER_PROJECTOR_PATTERN_CREATION_FAILURE,

	SCANNER_PROJECTION_TEST_DETECTION_FAILURE

} SCANNER_RESULT;


/***************************************************************************************
*	typedef struct FrameProperties
*	
* Description
*	contains image info, parameters retreived from camera
* Remarks
*
***************************************************************************************/
typedef struct FrameProperties {

	unsigned int frameNumber;		// camera sequence number
	int rows;
	int cols;
	unsigned long seconds;			// camera timestamp values
	unsigned long microSeconds;
	unsigned long cycleSeconds;
	unsigned long cycleCount;
	unsigned long cycleOffset;

	float frameRate;				// camera parameters		
	float shutter;					
	float gain;
	float brightness;
	float autoExposure;
	float sharpness;
	float whiteBalance;
	float hue;

	float saturation;
	float gamma;
	float iris;
	float focus;
	float zoom;
	float pan;
	float tilt;
	float triggerDelay;

	double currTimeQ;				// scanner timestamp values
	double prevTimeQ;
	double frequencyQ;
	double scannerFrequency;

	double d1;						// reserved for future
	double d2;
	double d3;
	double d4;

} FrameProperties;


/***************************************************************************************
*	typedef struct ImageData_GIPHeader
*	
* Description
*	standard header for GIP files
* Remarks
*	
***************************************************************************************/
typedef struct ImageData_GIPHeader {

	// First two fields must remain in place (and order), for handling both old and new file formats.
	bool isFiltered;		// True if filtered, false if raw.
	int frameNumber;		// 16 LSB contain frame number. 
							// 16 MSB determine old/new format. If 0 - old format, otherwise new format.

	// Header size + 4 reserved fields
	int headerSize;			// Total header size in bytes
	int headerReserved1;	// Reserved fields	
	int headerReserved2;
	int headerReserved3;	// size (in bytes) of CalibrationParameters
	int headerReserved4;

	// Size and format of reserved fields
	int reservedISize;		// Number of int elememts in the reservedI field.
	int reservedFSize;		// Number of float elememts in the reservedF field.
    int reservedDSize;		// Number of double elememts in the reservedD field.
	int reservedUISize;		// Number of unsigned int elememts in the reservedUI field.
	int reservedIFormat;	// Format of the reservedI field.
	int reservedFFormat;	// Format of the reservedF field.
    int reservedDFormat;	// Format of the reservedD field.
	int reservedUIFormat;	// Format of the reservedUI field.

	// 3D resolution
	int rows;
	int cols;
	
	// Texture
	int rowsTexture;
	int colsTexture;
	int bytesPerPixel;		// Default: 1
	int pixelFormat;		// Default: 1 (internal arrangement of bytes in each pixel)
	int numTextures;		// Default: 1
    int textureSequence;	// Default: 1 (sequence of texture patterns)
							//	200 - two synched images in stereo context
	
	// Frame properties
	int hasFrameProperties;		// 0 - no, 1 - yes
	int typeFrameProperties;	// currently not used
		
	// Time stamp
	int year;
	int month;
	int day;
	int hour;
	int minute;
	int second;
	int mSecond;	
	 
} ImageData_GIPHeader;


/***************************************************************************************
*	typedef struct CalibrationParameters
*
*  note: any change in this structure -> MUST also modify the following functions:
*		readCalibrationFile()
*		writeCalibrationFile()
*		readCalibrationParameters()
*		writeCalibrationParameters()
*		allocateCalibrationParameters()
*		clearCalibrationParameters()
*		copyCalibrationParameters()
*		sizeOfCalibrationParameters()
*		definition of this structure in C# - ImportedTypes.cs in GIPClientViewer solution
***************************************************************************************/
typedef struct CalibrationParameters {
	int calibFormat;		// 1 - normal usage
	int reserved1;
	int reserved2;
	
	int C_MatrixSize;		// size of C1,C2  - 3x4 matrices, 12 elements
	int RD1_VectorSize;		// size of RD1,RD2  - 1x6 vectors, 6 elements
	int RD2_VectorSize;
	int K_MatrixSize;		// size of K1,K2 - 3x3 matrices, 9 elements
	int RT_MatrixSize;		// size of RT1,RT2 - 3x4 matrices, 12 elements

	int reservedISize;
	int reservedFSize;
	int reservedUCSize;
	int reservedVSize;
	int reservedVBytesPerElement;

	float* C1;				// first camera matrix 
	float* C2;				// second camera matrix (or projector matrix)
	
	float* RD1;				// radial distortion parameters, first camera
	float* RD2;				// radial distortion parameters, second camera (or projector)

	float* K1;				// internal camera matrices
	float* K2;
	float* RT1;				// RT matrices for canonical form
	float* RT2;
	//float* C1_Canonical;	// C1_Canonical = K1 * RT1
	//float* C2_Canonical;	// C2_Canonical = K2 * RT2

	int* reservedI; 
	float* reservedF;
	unsigned char* reservedUC;
	void* reservedV;

} CalibrationParameters;


/***************************************************************************************
*	typedef struct ImageData_GIP
*
* header - standard GIP header. Contains size and format of all other fields.
* reserved fields - for specific user defined information. can be NULL.
* calibration parameters. can be NULL.
* x, y, z - 3D coordinates. can be NULL.
* w - fidelity (of 3D coordinates). can be NULL.
* texture - 2D data
* properties - camera properties 
***************************************************************************************/
typedef struct ImageData_GIP {
	ImageData_GIPHeader header;		// standard header
	int* reservedI;					// reserved fields
	float* reservedF;
	double* reservedD; 
	unsigned int* reservedUI;
	float* x;						// 3D coordinates
	float* y;
	float* z;
	float* w;						// raw fidelity
	unsigned char* texture;			// texture patterns
	FrameProperties* properties;	// frame properties for each texture
} ImageData_GIP;



/***************************************************************************************
* 
*
* Description
*
* Arguments
*
* Return value
*
* Remarks
*
***************************************************************************************/

extern "C" {

PROJECT_API void clearImageDataGIP(ImageData_GIP* image);
PROJECT_API SCANNER_RESULT initGIPHeader(ImageData_GIPHeader* header, const char* config);
PROJECT_API SCANNER_RESULT copyGIPHeader(ImageData_GIPHeader* source, ImageData_GIPHeader* target);
PROJECT_API SCANNER_RESULT copyReservedFields(ImageData_GIP* source, ImageData_GIP* target);
PROJECT_API SCANNER_RESULT allocateImageData(ImageData_GIP* data);
PROJECT_API SCANNER_RESULT copyImageData(ImageData_GIP* source, ImageData_GIP* target, int mode);

PROJECT_API int getSingleFrameSize(ImageData_GIPHeader* header);
PROJECT_API SCANNER_RESULT readFileGIP(const char* fileName, ImageData_GIP* data, int frameNumber, int mode);
PROJECT_API SCANNER_RESULT writeFileGIP(const char* fileName, ImageData_GIP* data, int frameNumber,	int mode);

PROJECT_API SCANNER_RESULT writeVRML(float* X, float* Y, float* Z, int rows, int cols, unsigned char* Texture, int rowsT, int colsT, char* outFileName, int vrmlType, bool isOldFormat);

PROJECT_API SCANNER_RESULT readSLCalibrationFile(const char* calibrationFileName, float C[3][4], float P[2][4], float RD[6]);

PROJECT_API SCANNER_RESULT readCalibrationFile(const char* calibrationFileName, CalibrationParameters* calibParams);
PROJECT_API SCANNER_RESULT writeCalibrationFile(const char* calibrationFileName, CalibrationParameters* calibParams);
PROJECT_API SCANNER_RESULT allocateCalibrationParameters(CalibrationParameters* calibParams);
PROJECT_API void clearCalibrationParameters(CalibrationParameters* calibParams);
PROJECT_API int sizeOfCalibrationParameters(CalibrationParameters* calibParams);
PROJECT_API SCANNER_RESULT copyCalibrationParameters(CalibrationParameters* src, CalibrationParameters* trg);
PROJECT_API SCANNER_RESULT readCalibrationParameters(FILE* f, CalibrationParameters* calib);
PROJECT_API SCANNER_RESULT writeCalibrationParameters(FILE* f, CalibrationParameters* calib);
PROJECT_API SCANNER_RESULT readCalibrationParametersFromGipFile(const char* fileName, CalibrationParameters* calibParams, ImageData_GIP* data);
PROJECT_API SCANNER_RESULT writeCalibrationParametersToGipFile(const char* fileName, CalibrationParameters* calibParams, ImageData_GIP* data);


PROJECT_API SCANNER_RESULT closeFileGIP(const char* fileName, int numFrames);

}

#endif


