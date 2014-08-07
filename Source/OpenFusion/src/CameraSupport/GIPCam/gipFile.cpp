/****************************************************
* gipFile.cpp										*
*                                                   *
* (C) GIP Laboratory								*
*     Computer Science Faculty, Technion.	2007	*
****************************************************/


#include "gipFile.h"


int FileSize(const char* fileName) {
	struct stat fileStat;
	int err = stat(fileName, &fileStat);
	if (0 != err) return 0;
	return fileStat.st_size;
}

/***************************************************************************************
* clearImageDataGIP
*
* Description
*	Frees all memory stored by the specified ImageData_GIP.
* 
* Arguments
*	data - pointer to the data structure to be cleared.
*
* Return value
*	None
*
* Remarks
*	Memory areas pointed by the fields of the specified ImageData_GIP are cleared.
*	data itself is not freed.
***************************************************************************************/
void clearImageDataGIP(ImageData_GIP* data) {

	if (!data) {
		return;
	}
	if (data->reservedI) {
		free(data->reservedI);
		data->reservedI = NULL;
	}
	if (data->reservedF) {
		free(data->reservedF);
		data->reservedF = NULL;
	}
	if (data->reservedD) {
		free(data->reservedD);
		data->reservedD = NULL;
	}
	if (data->reservedUI) {
		free(data->reservedUI);
		data->reservedUI = NULL;
	}
	if (data->x) {
		free(data->x);
		data->x = NULL;
	}
	if (data->y) {
		free(data->y);
		data->y = NULL;
	}
	if (data->z) {
		free(data->z);
		data->z = NULL;
	}
	if (data->w) {
		free(data->w);
		data->w = NULL;
	}
	if (data->texture) {
		free(data->texture);
		data->texture = NULL;
	}
	if (data->properties) {
		free(data->properties);
		data->properties = NULL;
	}
}


/***************************************************************************************
* copyGIPHeader
*
* Description
*	Copies all fields from the specified source structure to the specified target stucture.
*
* Arguments
*	source - pointer to the source ImageData_GIPHeader
*	target - pointer to the target ImageData_GIPHeader
*
* Return value
*	SCANNER_INVALID_ARGUMENTS if null arguments, SCANNER_OK otherwise.
*
* Remarks
*
***************************************************************************************/
SCANNER_RESULT copyGIPHeader(ImageData_GIPHeader* source, ImageData_GIPHeader* target) {
	
	if (!source || !target) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	// General Info
	target->isFiltered = source->isFiltered;
	target->frameNumber = source->frameNumber;
	target->headerSize = source->headerSize;
	target->headerReserved1 = source->headerReserved1;
	target->headerReserved2 = source->headerReserved2;
	target->headerReserved3 = source->headerReserved3;
	target->headerReserved4 = source->headerReserved4;
	target->reservedISize = source->reservedISize;
	target->reservedFSize = source->reservedFSize;
	target->reservedDSize = source->reservedDSize;
	target->reservedUISize = source->reservedUISize;
	target->reservedIFormat = source->reservedIFormat;
	target->reservedFFormat = source->reservedFFormat;
	target->reservedDFormat = source->reservedDFormat;
	target->reservedUIFormat = source->reservedUIFormat;

	// 3D resolution
	target->rows = source->rows;
	target->cols = source->cols;
	
	// Texture
	target->rowsTexture = source->rowsTexture;
	target->colsTexture = source->colsTexture;
	target->bytesPerPixel = source->bytesPerPixel;
	target->pixelFormat = source->pixelFormat;
	target->numTextures = source->numTextures;
	target->textureSequence = source->textureSequence;

	// Frame properties
	target->hasFrameProperties = source->hasFrameProperties;
	target->typeFrameProperties = source->typeFrameProperties;

	// Time stamp
	target->year = source->year;
	target->month = source->month;
	target->day = source->day;
	target->hour = source->hour;
	target->minute = source->minute;
	target->second = source->second;
	target->mSecond = source->mSecond;

	return SCANNER_OK;
}


SCANNER_RESULT initGIPHeader(ImageData_GIPHeader* header, const char* config) {
	if (!header) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	header->isFiltered = false;
	header->frameNumber = HEADER_OFFSET_SHIFT;
	header->headerSize = DEFAULT_HEADER_SIZE;
	header->headerReserved1 = 10;	
	header->headerReserved2 = 16;
	header->headerReserved3 = 0;
	header->headerReserved4 = 0;		
	header->reservedISize = 0;
	header->reservedFSize = 0;
    header->reservedDSize = 0;
	header->reservedUISize = 0;
	header->reservedIFormat = 0;
	header->reservedFFormat = 0;
    header->reservedDFormat = 0;
	header->reservedUIFormat = 0;
	header->rows = 0;
	header->cols = 0;
	header->rowsTexture = 0;
	header->colsTexture = 0;
	header->bytesPerPixel = 1;
	header->pixelFormat = 1;
	header->numTextures = 0;	
    header->textureSequence = 0;	
	header->hasFrameProperties = 0;		
	header->typeFrameProperties = 0;
	header->year = 0;
	header->month = 0;
	header->day = 0;
	header->hour = 0;
	header->minute = 0;
	header->second = 0;
	header->mSecond = 0;

	if (strcmp(config,"XYZRGB_360_480") == 0) {
		header->rows = 360;
		header->cols = 480;
		header->rowsTexture = 360;
		header->colsTexture = 480;
		header->bytesPerPixel = 1;
		header->pixelFormat = 3;
		header->numTextures = 3;	
		header->textureSequence = 13;	// TEMP - TODO
	}
	else if (strcmp(config,"XYZ+12PATTERNS") == 0) {
		header->rows = 360;
		header->cols = 480;
		header->rowsTexture = 360;
		header->colsTexture = 480;
		header->bytesPerPixel = 1;
		header->pixelFormat = 1;
		header->numTextures = 12;	
		header->textureSequence = 1;	
	}
	else if (strcmp(config,"HV_24_PATTERNS_360_480") == 0) {
		header->rowsTexture = 360;
		header->colsTexture = 480;
		header->bytesPerPixel = 1;
		header->pixelFormat = 1;
		header->numTextures = 24;	
		header->textureSequence = 5;	
	}
	else if (strcmp(config,"HV_24_PATTERNS_480_640") == 0) {
		header->rowsTexture = 480;
		header->colsTexture = 640;
		header->bytesPerPixel = 1;
		header->pixelFormat = 1;
		header->numTextures = 24;	
		header->textureSequence = 5;	
	}	
	else if (strcmp(config,"TEST_PATTERNS_360_480") == 0) {
		// TODO - (this is only temp test patterns)
		header->rowsTexture = 360;
		header->colsTexture = 480;
		header->bytesPerPixel = 1;
		header->pixelFormat = 1;
		header->numTextures = 12;	
		header->textureSequence = 1;	
	}
	else {
		return SCANNER_INVALID_ARGUMENTS;
	}

	return SCANNER_OK;
}


/***************************************************************************************
* copyReservedFields
*
* Description
*	Copies all reserved fields from the specified source structure to the specified target stucture.
*
* Arguments
*	source - pointer to the source ImageData_GIP
*	target - pointer to the target ImageData_GIP
*
* Return value
*	SCANNER_INVALID_ARGUMENTS if null arguments, SCANNER_OK otherwise.
*
* Remarks
*
***************************************************************************************/
SCANNER_RESULT copyReservedFields(ImageData_GIP* source, ImageData_GIP* target) {
	if (!source || !target) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	int reservedISize = source->header.reservedISize;
	if (reservedISize > 0) {
		memcpy(target->reservedI, source->reservedI, reservedISize * sizeof(int));
	}
	int reservedFSize = source->header.reservedFSize;
	if (reservedFSize > 0) {
		memcpy(target->reservedF, source->reservedF, reservedFSize * sizeof(float));
	}
	int reservedDSize = source->header.reservedDSize;
	if (reservedDSize > 0) {
		memcpy(target->reservedD, source->reservedD, reservedDSize * sizeof(double));
	}
	int reservedUISize = source->header.reservedUISize;
	if (reservedUISize > 0) {
		memcpy(target->reservedUI, source->reservedUI, reservedUISize * sizeof(unsigned int));
	}

	return SCANNER_OK;
}



/***************************************************************************************
* copyFrameProperties
*
* Description
*	Copies all fields from the specified target structure to the specified source stucture.
*
* Arguments
*	source - pointer to the source FrameProperties
*	target - pointer to the target FrameProperties
*
* Return value
*	SCANNER_INVALID_ARGUMENTS if null arguments, SCANNER_OK otherwise.
*
* Remarks
*
***************************************************************************************/
SCANNER_RESULT copyFrameProperties(FrameProperties* source, FrameProperties* target) {

	if (!source || !target) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	target->frameNumber = source->frameNumber;
	target->rows = source->rows;
	target->cols = source->cols;
	target->seconds = source->seconds;
	target->microSeconds = source->microSeconds;
	target->cycleSeconds = source->cycleSeconds;
	target->cycleCount = source->cycleCount;
	target->cycleOffset = source->cycleOffset;

	target->frameRate = source->frameRate;
	target->shutter = source->shutter;
	target->gain = source->gain;
	target->brightness = source->brightness;
	target->autoExposure = source->autoExposure;
	target->sharpness = source->sharpness;
	target->whiteBalance = source->whiteBalance;
	target->hue = source->hue;

	target->saturation = source->saturation;
	target->gamma = source->gamma;
	target->iris = source->iris;
	target->focus = source->focus;
	target->zoom = source->zoom;
	target->pan = source->pan;
	target->tilt = source->tilt;
	target->triggerDelay = source->triggerDelay;
	
	target->currTimeQ = source->currTimeQ;
	target->prevTimeQ = source->prevTimeQ;
	target->frequencyQ = source->frequencyQ;
	target->scannerFrequency = source->scannerFrequency;

	target->d1 = source->d1;
	target->d2 = source->d2;
	target->d3 = source->d3;
	target->d4 = source->d4; 


	return SCANNER_OK;
}




/***************************************************************************************
* allocateImageData
*
* Description
*	Allocates memory to hold image data, according to size and format specified in
*	the data's header.
*
* Arguments
*	data - pointer to a structure used to store the data.
* 
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
*
* Remarks
*	Allocation of reserved fields, 3D data, 2D textures, 2D frame properties.
***************************************************************************************/
SCANNER_RESULT allocateImageData(ImageData_GIP* data) {	

	if (!data) {
		return SCANNER_INVALID_ARGUMENTS;
	}		
	
	// reserved areas
	if (data->header.reservedISize > 0) {
		data->reservedI = (int*)malloc((data->header.reservedISize) * sizeof(int));
		if (!(data->reservedI)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}	
	}
	else {
		data->reservedI = 0;
	}

	if (data->header.reservedFSize > 0) {
		data->reservedF = (float*)malloc((data->header.reservedFSize) * sizeof(float));
		if (!(data->reservedF)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}	
	}
	else {
		data->reservedF = 0;
	}

	if (data->header.reservedDSize > 0) {
		data->reservedD = (double*)malloc((data->header.reservedDSize) * sizeof(double));
		if (!(data->reservedD)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}	
	}
	else {
		data->reservedD = 0;
	}

	if (data->header.reservedUISize > 0) {
		data->reservedUI = (unsigned int*)malloc((data->header.reservedUISize) * sizeof(unsigned int));
		if (!(data->reservedUI)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}	
	}
	else {
		data->reservedUI = 0;
	}	

	// 3D data
	int numPoints = data->header.rows * data->header.cols;
	if (numPoints > 0) {
		data->x = (float*)malloc(numPoints * sizeof(float));
		data->y = (float*)malloc(numPoints * sizeof(float));
		data->z = (float*)malloc(numPoints * sizeof(float));		
		if (!(data->x) || !(data->y) || !(data->z)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!(data->header.isFiltered)) {
			data->w = (float*)malloc(numPoints * sizeof(float));
			if (!data->w) {
				clearImageDataGIP(data);
				return SCANNER_MEMORY_ALLOCATION_FAILURE;
			}
		}
		else {
			data->w = 0;
		}
	}
	else {
		data->x = 0;
		data->y = 0;
		data->z = 0;
		data->w = 0;
	}
	
	// Texture 
	int textureSize = data->header.numTextures * data->header.bytesPerPixel * data->header.rowsTexture * data->header.colsTexture;	
	if (textureSize > 0) {
		data->texture = (unsigned char*)malloc(textureSize * sizeof(unsigned char));
		if (!(data->texture)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
	}
	else {
		data->texture = 0;
	}

	// Frame properties
	if (data->header.hasFrameProperties == 1) {
		data->properties = (FrameProperties*)malloc((data->header.numTextures) * sizeof(FrameProperties));
		if (!(data->properties)) {
			clearImageDataGIP(data);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
	}
	else {
		data->properties = 0;
	}

	return SCANNER_OK;
}



/***************************************************************************************
* copyImageData
*
* Description
*	Copies data from specified source to target ImageData_GIP structures.
* Arguments
*	source - pointer to the source location
*	target - pointer to the target location
*	mode - mode of operation:
*			1 - copy header only (no data is written).
*			2 - copy header + reserved fields only (no data is written).
*			Otherwise - header + reserved + data are copied.
* Return value
*	SCANNER_OK if successful, error indication otherwise.
* Remarks
*
*	TODO
*	does not check validity of field sizes
*	!!! assumes input holds 1 frame - not good for video 
***************************************************************************************/
SCANNER_RESULT copyImageData(ImageData_GIP* source, ImageData_GIP* target, int mode) {

	SCANNER_RESULT res;

	if (!source || !target) {
		return SCANNER_INVALID_ARGUMENTS;
	}
	
	res = copyGIPHeader(&(source->header), &(target->header));
	if (res != SCANNER_OK) {
		return res;
	}
	if (mode == 1) {
		return SCANNER_OK;
	}

	// reserved fields
	int reservedISize = source->header.reservedISize;
	if (reservedISize > 0) {
		memcpy(target->reservedI, source->reservedI, reservedISize * sizeof(int));
	}
	int reservedFSize = source->header.reservedFSize;
	if (reservedFSize > 0) {
		memcpy(target->reservedF, source->reservedF, reservedFSize * sizeof(float));
	}
	int reservedDSize = source->header.reservedDSize;
	if (reservedDSize > 0) {
		memcpy(target->reservedD, source->reservedD, reservedDSize * sizeof(double));
	}
	int reservedUISize = source->header.reservedUISize;
	if (reservedUISize > 0) {
		memcpy(target->reservedUI, source->reservedUI, reservedUISize * sizeof(unsigned int));
	}

	if (mode == 2) {
		return SCANNER_OK;
	}	

	// 3D data
	int numPoints = source->header.rows * source->header.cols;	
	if (numPoints > 0) {
		memcpy(target->x, source->x, numPoints * sizeof(float));
		memcpy(target->y, source->y, numPoints * sizeof(float));
		memcpy(target->z, source->z, numPoints * sizeof(float));
		if (!source->header.isFiltered) {
			memcpy(target->w, source->w, numPoints * sizeof(float));
		}
	}
	
	// textures
	int textureSize = source->header.numTextures * source->header.bytesPerPixel * source->header.rowsTexture * source->header.colsTexture;
	if (textureSize > 0) {
		memcpy(target->texture, source->texture, textureSize * sizeof(unsigned char));
	}
	
	// Frame properties
	if (source->header.hasFrameProperties) {
		memcpy(target->properties, source->properties, (source->header.numTextures) * sizeof(FrameProperties));
		/*
		res = copyFrameProperties(image->properties, data->properties);
		if (res != SCANNER_OK) {
			return res;
		}*/
	}

	return SCANNER_OK;
}



/***************************************************************************************
* readGIPFileHeader
*
* Description
*	Reads file header and stores info in the specified data structure.
* 
* Arguments
*	f - input file
*	data - pointer to ImageData_GIP structure in which to store the data from file.
*	isCorruptedFile - files having 12 textures but header states only one texture
*
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* 
* Remarks
*	The file should already be opened and will be closed later. 
***************************************************************************************/
SCANNER_RESULT readGIPFileHeader(FILE* f, ImageData_GIP* data, bool isCorruptedFile) {
	
	ImageData_GIPHeader* header = &(data->header);
	int headerOffset = 0;
	int frameNumberField = 0;

	if (!fread(&(header->isFiltered), sizeof(bool), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	if (!fread(&frameNumberField, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_READ_FAILURE;
	}
	// Check headerOffset to determine which format is used.
	headerOffset = frameNumberField / HEADER_OFFSET_SHIFT;
	

	if (headerOffset == 0) {
		// Old file format
		
		// Number of frames
		header->frameNumber = frameNumberField;

		// Read rows, cols (of 3D resolution)
		if (!fread(&(header->rows), sizeof(int), 1, f)) {
			return SCANNER_FILE_READ_FAILURE;
		}	
		if (!fread(&(header->cols), sizeof(int), 1, f)) {
			return SCANNER_FILE_READ_FAILURE;
		}

		// Fill default settings		
		//header->headerSize = sizeof(ImageData_GIPHeader);
		header->headerSize =  DEFAULT_HEADER_SIZE;

		header->headerReserved1 = 0;
		header->headerReserved2 = 0;
		header->headerReserved3 = 0;
		header->headerReserved4 = 0;
		header->reservedISize = 0;
		header->reservedFSize = 0;
		header->reservedDSize = 0;
		header->reservedUISize = 0;
		header->reservedIFormat = 0;
		header->reservedFFormat = 0;
		header->reservedDFormat = 0;
		header->reservedUIFormat = 0;

		header->rowsTexture = DEFAULT_ROWS_TEXTURE;
		header->colsTexture = DEFAULT_COLS_TEXTURE;
		header->bytesPerPixel = DEFAULT_BYTES_PER_PIXEL;
		header->pixelFormat = DEFAULT_PIXEL_FORMAT;
		header->numTextures = DEFAULT_NUM_TEXTURES;
		if (isCorruptedFile) {
			header->numTextures = 12;
		}
		header->textureSequence = DEFAULT_TEXTURE_SEQUENCE;

		header->hasFrameProperties = 0;
		header->typeFrameProperties = 0;
		
		header->year = 0;
		header->month = 0;
		header->day = 0;
		header->hour = 0;
		header->minute = 0;
		header->second = 0;
		header->mSecond = 0;

		// TODO - set zero or default values in all other fields
		
		return SCANNER_OK;
	}
	
	// New file format

	// Number of frames
	header->frameNumber = frameNumberField - HEADER_OFFSET_SHIFT;

	// Header size
	if (!fread(&(header->headerSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	//  4 Reserved fields
	if (!fread(&(header->headerReserved1), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->headerReserved2), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->headerReserved3), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->headerReserved4), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	// Size and format of reserved memory areas
	if (!fread(&(header->reservedISize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedFSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedDSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedUISize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedIFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedFFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedDFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->reservedUIFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	// 3D resolution
	if (!fread(&(header->rows), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->cols), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	
	// Texture
	if (!fread(&(header->rowsTexture), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->colsTexture), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->bytesPerPixel), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->pixelFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->numTextures), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->textureSequence), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}	

	// FrameProperties
	if (!fread(&(header->hasFrameProperties), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->typeFrameProperties), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	// Time stamp
	if (!fread(&(header->year), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->month), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->day), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->hour), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->minute), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->second), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(header->mSecond), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	
	return SCANNER_OK;
}





/***************************************************************************************
* getSingleFrameSize
*
* Description
*	Returns number of bytes in a single frame. 
* Arguments
*	header - ImageData_GIPHeader containing the relevant info.
* Return value
*	number of bytes in a single frame.
* Remarks
*	total of [x,y,z,w,texture,frameProperties] representation. 
***************************************************************************************/
int getSingleFrameSize(ImageData_GIPHeader* header) {
	int frameSize = 0;

	if (header->isFiltered) {
		// x,y,z
		frameSize = 3 * (header->rows) * (header->cols) * sizeof(float);
	}
	else {
		// x,y,z,w
		frameSize = 4 * (header->rows) * (header->cols) * sizeof(float);
	}

	frameSize += (header->numTextures) * (header->bytesPerPixel) * 
				(header->rowsTexture) * (header->colsTexture) * sizeof(unsigned char);
	
	if (header->hasFrameProperties == 1) {
		frameSize += (header->numTextures) * sizeof(FrameProperties);
	}

	return frameSize;
}



/***************************************************************************************
* readFrameProperties
*
* Description
*	Reads frame properties info from a file into a FrameProperties structure
* Arguments
*	f - input file
*	frameProperties - pointer to a FrameProperties structure in which to store the data
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* 
* Remarks
*	The file should already be opened and will be closed later. 
***************************************************************************************/
SCANNER_RESULT readFrameProperties(FILE* f, FrameProperties* frameProperties) {

	if (!fread(&(frameProperties->frameNumber), sizeof(unsigned int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->rows), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->cols), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	// camera timestamp
	if (!fread(&(frameProperties->seconds), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->microSeconds), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->cycleSeconds), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->cycleCount), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->cycleOffset), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}	

	// camera properties
	if (!fread(&(frameProperties->frameRate), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->shutter), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->gain), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->brightness), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->autoExposure), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->sharpness), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->whiteBalance), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->hue), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->saturation), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->gamma), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->iris), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->focus), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->zoom), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->pan), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->tilt), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->triggerDelay), sizeof(float), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	// scanner timestamp
	if (!fread(&(frameProperties->currTimeQ), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->prevTimeQ), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->frequencyQ), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->scannerFrequency), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	// reserved doubles
	if (!fread(&(frameProperties->d1), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->d2), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->d3), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(frameProperties->d4), sizeof(double), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	return SCANNER_OK;
}


bool shouldSwapFirstTwoTextures(unsigned char* texture) {
	
	int sum1 = 0;
	int sum2 = 0;
	int rowStep = 5;
	int colStep = 5;
	int p1, p2;

	for (int i=0; i < DEFAULT_ROWS_TEXTURE; i+= rowStep) {
		for (int j=0; j < DEFAULT_COLS_TEXTURE; j+= colStep) {
			p1 = i*DEFAULT_COLS_TEXTURE + j;
			p2 = p1 + DEFAULT_ROWS_TEXTURE * DEFAULT_COLS_TEXTURE;
			sum1 += texture[p1];
			sum2 += texture[p2];
		}
	}

	if (sum1 > sum2) {
		return true;
	}	
	return false;
}



/***************************************************************************************
* readFileGIP
*
* Description
*	Reads contents of a GIP standard file into a data structure.
*
* Arguments
*	fileName - input file name.
*	data - pointer to the ImageData_GIP structure in which to store the data.
*	frameNumber - index of the desired frame in the file, 0 based.
*	mode - currently not used.
*
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
*
* Remarks
*	Memory is allocated to the specified data, according to info in the file header.
*	This memory must be released by calling clearImageDataGIP().
*	
*	Calling this function with (frameNumber == 0) will fill the ImageData_GIP structure
*	with header fields, reserved fields, and data of the 1st frame. The header specifies the
*	number of frames (which is 1, except in videos). 
*	Calling this function with a valid positive frameNumber will fill the ImageData_GIP structure
*	with (the same) header and reserved fields, and data of the N'th frame (0 based).
***************************************************************************************/
SCANNER_RESULT readFileGIP(const char* fileName, ImageData_GIP* data, int frameNumber, int mode) {

	FILE* f = NULL;
	errno_t err;
	SCANNER_RESULT res;	
	bool isCorruptedFile = false;

	if (!data) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	err = fopen_s(&f, fileName, "rb");
	if (err != 0) {
		return SCANNER_FILE_OPEN_FAILURE;
	}
	
	//printf("** Opening: %s\n", fileName);

	// check if file format is corrupted
	int fileSize = FileSize(fileName);
	if (fileSize == CORRUPTED_FILE_SIZE) {
		isCorruptedFile = true;
	}

	// Read file header
	res = readGIPFileHeader(f, data, isCorruptedFile);
	if (res != SCANNER_OK) {
		fclose(f);
		return res;
	}

	// Read reserved fields if existing
	int reservedISize = data->header.reservedISize;
	if (reservedISize > 0) {
		data->reservedI = (int*)malloc(reservedISize * sizeof(int));
		if (!(data->reservedI)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(data->reservedI, reservedISize * sizeof(int), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
	}
	else {
		data->reservedI = NULL;
	}

	int reservedFSize = data->header.reservedFSize;
	if (reservedFSize > 0) {
		data->reservedF = (float*)malloc(reservedFSize * sizeof(float));
		if (!(data->reservedF)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(data->reservedF, reservedFSize * sizeof(float), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
	}
	else {
		data->reservedF = NULL;
	}
	
	int reservedDSize = data->header.reservedDSize;
	if (reservedDSize > 0) {
		data->reservedD = (double*)malloc(reservedDSize * sizeof(double));
		if (!(data->reservedD)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(data->reservedD, reservedDSize * sizeof(double), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
	}
	else {
		data->reservedD = NULL;
	}

	int reservedUISize = data->header.reservedUISize;
	if (reservedUISize > 0) {
		data->reservedUI = (unsigned int*)malloc(reservedUISize * sizeof(unsigned int));
		if (!(data->reservedUI)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(data->reservedUI, reservedUISize * sizeof(unsigned int), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
	}
	else {
		data->reservedUI = NULL;
	}	

	// Skip to specified frame number	
	__int64 skipSize = 0;
	if (data->header.headerReserved3 > 0) {
		skipSize = (__int64)data->header.headerReserved3;
	}
	if (frameNumber > 0) {
		int frameSize = getSingleFrameSize(&(data->header));
		skipSize = skipSize + (__int64)frameSize * (__int64)frameNumber;		
	}
	if (_fseeki64(f, skipSize ,SEEK_CUR)) {
		clearImageDataGIP(data);
		fclose(f);
		return SCANNER_FILE_READ_FAILURE;
	}

	// Read coordinates and textures.
	int numPoints = data->header.rows * data->header.cols;
	if (numPoints > 0) {
		// Allocate memory for x, y, z
		data->x = (float*)malloc(numPoints * sizeof(float));
		data->y = (float*)malloc(numPoints * sizeof(float));
		data->z = (float*)malloc(numPoints * sizeof(float));
		if (!(data->x) || !(data->y) || !(data->z)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}

		// Read x, y, z
		if (!fread(data->x, numPoints * sizeof(float), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
		if (!fread(data->y, numPoints * sizeof(float), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
		if (!fread(data->z, numPoints * sizeof(float), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
		// Read raw fidelity (W) if existing
		if (!(data->header.isFiltered)) {
			data->w = (float*)malloc(numPoints * sizeof(float));
			if (!data->w) {
				clearImageDataGIP(data);
				fclose(f);
				return SCANNER_MEMORY_ALLOCATION_FAILURE;
			}	
			if (!fread(data->w, numPoints * sizeof(float), 1, f)) {
				clearImageDataGIP(data);
				fclose(f);
				return SCANNER_FILE_READ_FAILURE;
			}
		}
		else {
			data->w = NULL;
		}
	}
	else {
		data->x = NULL;
		data->y = NULL;
		data->z = NULL;
		data->w = NULL;
	}
		
	// Read textures
	int textureSize = data->header.numTextures * data->header.bytesPerPixel * data->header.rowsTexture * data->header.colsTexture;	
	if (textureSize > 0) {
		data->texture = (unsigned char*)malloc(textureSize * sizeof(unsigned char));
		if (!(data->texture)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(data->texture, textureSize * sizeof(unsigned char), 1, f)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
		// swap first two textures if the first one is brighter
		//		only done in some cases where original bright and dark camera images were swapped
		//		and the accumulated data contains 12 or 15 images of 480x640 pixels
		
		/*if ((data->header.numTextures == 12 || data->header.numTextures == 15) 
			&& data->header.rowsTexture == 480 && data->header.colsTexture == 640) {
			if (shouldSwapFirstTwoTextures(data->texture)) {
				printf("** Swapped textures in: %s\n", fileName);
				int singleTextureSize = 480 * 640;
				unsigned char* tempT = (unsigned char*)malloc(singleTextureSize * sizeof(unsigned char));
				memcpy(tempT, data->texture, singleTextureSize);
				memcpy(data->texture, data->texture + singleTextureSize, singleTextureSize);
				memcpy(data->texture + singleTextureSize, tempT, singleTextureSize);
				free(tempT);
			}
			
		}*/
	}
	else {
		data->texture = NULL;
	}

	// Read frame properties
	if (data->header.hasFrameProperties == 1) {
		data->properties = (FrameProperties*)malloc(data->header.numTextures * sizeof(FrameProperties));
		if (!(data->properties)) {
			clearImageDataGIP(data);
			fclose(f);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		for (int i = 0; i < data->header.numTextures; i++) {
			res = readFrameProperties(f, (data->properties) + i);
			if (res != SCANNER_OK) {
				clearImageDataGIP(data);
				fclose(f);
				return res;
			}
		}		
	}
	else {
		data->properties = NULL;

	}

	fclose(f);
	return SCANNER_OK;
	
}
	

/***************************************************************************************
* writeGIPFileHeader
*
* Description
*	Writes file header, from the specified data. 
* Arguments
*	f - output file
*	data - pointer to ImageData_GIP structure holding the actual data.
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	The file should already be opened and will be closed later. 
***************************************************************************************/
SCANNER_RESULT writeGIPFileHeader(FILE* f, ImageData_GIP* data) {

	ImageData_GIPHeader* header = &(data->header);
	int frameNumberField;	// Contains both frame number and header offset.

	// General Info
	if (!fwrite(&(header->isFiltered), sizeof(bool), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}	
	// Write 'frameNumberField' containing:
	//		16 LSB - frame number
	//		16 MSB - file format. 0 - old file format, 1 - new file format. 
	frameNumberField = 1 * HEADER_OFFSET_SHIFT + (header->frameNumber); 
	if (!fwrite(&frameNumberField, sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}

	if (!fwrite(&(header->headerSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->headerReserved1), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->headerReserved2), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->headerReserved3), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->headerReserved4), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedISize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedFSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedDSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedUISize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedIFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedFFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedDFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->reservedUIFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}

	// 3D resolution
	if (!fwrite(&(header->rows), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->cols), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}

	// Texture
	if (!fwrite(&(header->rowsTexture), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->colsTexture), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->bytesPerPixel), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->pixelFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->numTextures), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->textureSequence), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}

	// Frame properties
	if (!fwrite(&(header->hasFrameProperties), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->typeFrameProperties), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}

	// Time stamp
	if (!fwrite(&(header->year), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->month), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->day), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->hour), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->minute), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->second), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	if (!fwrite(&(header->mSecond), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;		
	}
	
	return SCANNER_OK;
} 



/***************************************************************************************
* writeFrameProperties
*
* Description
*	Write frame properties data into a file.
* Arguments
*	f - output file
*	frameProperties - pointer to a FrameProperties structure containing the data
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	The file should already be opened and will be closed later.
***************************************************************************************/
SCANNER_RESULT writeFrameProperties(FILE* f, FrameProperties* frameProperties) {

	if (!fwrite(&(frameProperties->frameNumber), sizeof(unsigned int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->rows), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->cols), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}

	// camera timestamp
	if (!fwrite(&(frameProperties->seconds), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->microSeconds), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->cycleSeconds), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->cycleCount), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->cycleOffset), sizeof(unsigned long), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	
	// camera properties
	if (!fwrite(&(frameProperties->frameRate), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->shutter), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->gain), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->brightness), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->autoExposure), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->sharpness), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->whiteBalance), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->hue), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->saturation), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->gamma), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->iris), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->focus), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->zoom), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->pan), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->tilt), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->triggerDelay), sizeof(float), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}

	// scanner timestamp
	if (!fwrite(&(frameProperties->currTimeQ), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->prevTimeQ), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->frequencyQ), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->scannerFrequency), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}

	// reserved doubles
	if (!fwrite(&(frameProperties->d1), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->d2), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->d3), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(frameProperties->d4), sizeof(double), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}

	return SCANNER_OK;
}




/***************************************************************************************
* writeFileGIP()
*
* Description
*	Writes to file from specified data structure
* Arguments
*	fileName - output file name.
*	data - pointer to ImageData_GIP structure holding the actual data.
*	frameNumber - 0 if single shot or first among video	frames.
*				Otherwise the video frame number.
*				If not zero - only data is written.
*	mode - mode of operation:
*			applied only if frameNumber is zero. 
*			1 - write header only (no data is written).
*			2 - write header + reserved fields only (no data is written).
*			Otherwise - data is written.
*
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	Header and reserved fields will be written only if frameNumber == 0
*	If frameNumber is 0, the file is opened in "wb" mode. (deletes file if existing) 
*	Otherwise, it is opened in "a+b" mode. (appends data, and doesn't delete file).
*	 
***************************************************************************************/
SCANNER_RESULT writeFileGIP(const char* fileName, ImageData_GIP* data, int frameNumber,	int mode) {

	FILE* f;
	errno_t err;
	SCANNER_RESULT res;
	
	if (!data) {
		return SCANNER_INVALID_ARGUMENTS;
	}
	
	if (frameNumber == 0) {
		err = fopen_s(&f, fileName, "wb");
	}
	else {
		err = fopen_s(&f, fileName, "a+b");
	}
		
	if (err != 0) {
		return SCANNER_FILE_OPEN_FAILURE;
	}
	
	if (frameNumber == 0) {
		// Write header and reserved fields only once. 
		res = writeGIPFileHeader(f, data);
		if (res != SCANNER_OK) {
			fclose(f);
			return res;
		}

		if (mode == 1) {
			fclose(f);
			return SCANNER_OK;			
		}
	
		// Write reserved fields if existing
		int reservedISize = data->header.reservedISize;
		if (reservedISize > 0) {
			if (!fwrite(data->reservedI, reservedISize * sizeof(int), 1, f)) {
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}
		}
	
		int reservedFSize = data->header.reservedFSize;
		if (reservedFSize > 0) {
			if (!fwrite(data->reservedF, reservedFSize * sizeof(float), 1, f)) {
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}
		}
		
		int reservedDSize = data->header.reservedDSize;
		if (reservedDSize > 0) {
			if (!fwrite(data->reservedD, reservedDSize * sizeof(double), 1, f)) {
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}
		}

		int reservedUISize = data->header.reservedUISize;
		if (reservedUISize > 0) {
			if (!fwrite(data->reservedUI, reservedUISize * sizeof(unsigned int), 1, f)) {
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}
		}

		if (mode == 2) {
			fclose(f);
			return SCANNER_OK;			
		}	

		// skip calibration parameters
		/*
		__int64 skipSize = (__int64)data->header.headerReserved3;
		if (skipSize > 0) {
			if (_fseeki64(f, skipSize, SEEK_CUR)) {
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}
		}
		*/
		// write "dummy" calibration, real one will be written later
		if (data->header.headerReserved3 > 0) {
			byte* dummy = (byte*)malloc(data->header.headerReserved3 * sizeof(byte));
			if (!fwrite(dummy, data->header.headerReserved3 * sizeof(byte), 1, f)) {
				free(dummy);
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}
			free(dummy);
		}

	}
	else {
		int frameSize = getSingleFrameSize(&(data->header));
		
		//int headerSize = sizeof(ImageData_GIPHeader) - 3; 
		int headerSize = data->header.headerSize;

		/*
		int skipSize = headerSize
						+ (data->header.reservedISize) * sizeof(int)
						+ (data->header.reservedFSize) * sizeof(float)
						+ (data->header.reservedDSize) * sizeof(double)
						+ (data->header.reservedUISize) * sizeof(unsigned int)
						+ frameSize * frameNumber;
		if (fseek(f, skipSize, SEEK_SET)) {
			fclose(f);
			return SCANNER_FILE_WRITE_FAILURE;
		}
		*/
		__int64 skipSize = headerSize
							+ (data->header.reservedISize) * sizeof(int)
							+ (data->header.reservedFSize) * sizeof(float)
							+ (data->header.reservedDSize) * sizeof(double)
							+ (data->header.reservedUISize) * sizeof(unsigned int)
							+ (__int64)frameSize * (__int64)frameNumber;
		
		if (data->header.headerReserved3 > 0) {
			// add size of CalibrationParameters to skipSize
			skipSize += (__int64)data->header.headerReserved3;
		}

		if (_fseeki64(f, skipSize, SEEK_SET)) {
			fclose(f);
			return SCANNER_FILE_WRITE_FAILURE;
		}
		
	}
	

	// Write coordinates.
	int numPoints = data->header.rows * data->header.cols;	
	if (numPoints > 0) {
		if (!fwrite(data->x, numPoints * sizeof(float), 1, f)) {
			fclose(f);
			return SCANNER_FILE_WRITE_FAILURE;
		}
		if (!fwrite(data->y, numPoints * sizeof(float), 1, f)) {
			fclose(f);
			return SCANNER_FILE_WRITE_FAILURE;
		}
		if (!fwrite(data->z, numPoints * sizeof(float), 1, f)) {
			fclose(f);
			return SCANNER_FILE_WRITE_FAILURE;
		}
		if (!(data->header.isFiltered)) {	
			if (!fwrite(data->w, numPoints * sizeof(float), 1, f)) {
				fclose(f);
				return SCANNER_FILE_WRITE_FAILURE;
			}	
		}
	}

	// Write textures.	
	int textureSize = data->header.numTextures * data->header.bytesPerPixel * data->header.rowsTexture * data->header.colsTexture;
	if (textureSize > 0) {
		if (!fwrite(data->texture, textureSize * sizeof(unsigned char), 1, f)) {
			fclose(f);
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	// Write frame properties
	if (data->header.hasFrameProperties == 1) {
		for (int i = 0; i < data->header.numTextures; i++) {
			res = writeFrameProperties(f, (data->properties) + i);
			if (res != SCANNER_OK) {
				clearImageDataGIP(data);
				fclose(f);
				return res;
			}
		}		
	}
	
	fclose(f);
	return SCANNER_OK;
}


/***************************************************************************************
* closeFileGIP()
*
* Description
*	writes number of frames and current date and time to a file 
* Arguments
*
* Return value
*
* Remarks
*
***************************************************************************************/
SCANNER_RESULT closeFileGIP(const char* fileName, int numFrames) {
	FILE* f;
	errno_t err;	
	
	if (numFrames <= 0) {
		return SCANNER_INVALID_ARGUMENTS;
	}
	
	err = fopen_s(&f, fileName, "r+b");
	if (err != 0) {
		return SCANNER_FILE_OPEN_FAILURE;
	}

	// skip 1 byte and write numFrames field
	int skipSize = 1;
	if (fseek(f, skipSize, SEEK_SET)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}

	int frameNumberField = HEADER_OFFSET_SHIFT + numFrames;	
	if (!fwrite(&frameNumberField, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}

	// close and reopen file
	fclose(f);	
	err = fopen_s(&f, fileName, "r+b");
	if (err != 0) {
		return SCANNER_FILE_OPEN_FAILURE;
	}

	// skip 97 bytes and write timestamp fields
	if (fseek(f, 97, SEEK_SET)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}

	SYSTEMTIME st;    	
	GetLocalTime(&st);
	int val;

	val = (int)st.wYear;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}
	val = (int)st.wMonth;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}
	val = (int)st.wDay;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}
	val = (int)st.wHour;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}
	val = (int)st.wMinute;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}
	val = (int)st.wSecond;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}
	val = (int)st.wMilliseconds;
	if (!fwrite(&val, sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}

	fclose(f);
	return SCANNER_OK;
}



/***************************************************************************************
* readSLCalibrationFile()
*
* Description
*	reads calibration file
* Arguments
*	C - the 3x4 camera calibration matrix
*	P - a 2x4 submatrix of the projector calibration matrix, with one line omitted, since it is not needed for the reconstruction
*	RD - the radial distortion parameters of the camera 
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	this function supports calibration file written by MATLAB calibration process intended for reconstruction from grey-coded patterns.
*	!!! use only for reading "calibration_CB.dat" having size 208 or 256 bytes 
*	!!! use the readCalibrationFile() function in all other cases
***************************************************************************************/
SCANNER_RESULT readSLCalibrationFile(const char* calibrationFileName, float C[3][4], float P[2][4], float RD[6]) {
	int i;
	int j;
	FILE* f;
	errno_t err;
	double Cd[4][3];
	double Pd[4][2];
	double RDd[6];		

	err = fopen_s(&f, calibrationFileName, "rb");
	if (err != 0) {
		return SCANNER_CALIBRATION_FILE_NOT_FOUND;
	}

	for (i=0; i<4; i++) {
		for (j=0; j<3; j++) {
			if (!(fread(&(Cd[i][j]), sizeof(double), 1, f))) {
				fclose(f);
				return SCANNER_FILE_READ_FAILURE;
			}
			C[j][i] = (float)Cd[i][j];
		}
	}
	for (i=0; i<4; i++) {
		for (j=0; j<2; j++) {
			if (!(fread(&(Pd[i][j]), sizeof(double), 1, f))) {
				fclose(f);
				return SCANNER_FILE_READ_FAILURE;
			}
			P[j][i] = (float)Pd[i][j];
		}
	}
	for (i=0; i<6; i++) {
		if (!(fread(&(RDd[i]), sizeof(double), 1, f))) {
			fclose(f);
			return SCANNER_FILE_READ_FAILURE;
		}
		RD[i] = (float)RDd[i];
	}
	fclose(f);

	return SCANNER_OK;
}


