
#include "gipFile.h"


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
void clearCalibrationParameters(CalibrationParameters* calibParams) {
	if (!calibParams) {
		return;
	}
	if (calibParams->C1) {
		free(calibParams->C1);
		calibParams->C1 = NULL;
	}
	if (calibParams->C2) {
		free(calibParams->C2);
		calibParams->C2 = NULL;
	}
	if (calibParams->RD1) {
		free(calibParams->RD1);
		calibParams->RD1 = NULL;
	}
	if (calibParams->RD2) {
		free(calibParams->RD2);
		calibParams->RD2 = NULL;
	}
	if (calibParams->K1) {
		free(calibParams->K1);
		calibParams->K1 = NULL;
	}
	if (calibParams->K2) {
		free(calibParams->K2);
		calibParams->K2 = NULL;
	}
	if (calibParams->RT1) {
		free(calibParams->RT1);
		calibParams->RT1 = NULL;
	}
	if (calibParams->RT2) {
		free(calibParams->RT2);
		calibParams->RT2 = NULL;
	}
	if (calibParams->reservedI) {
		free(calibParams->reservedI);
		calibParams->reservedI = NULL;
	}
	if (calibParams->reservedF) {
		free(calibParams->reservedF);
		calibParams->reservedF = NULL;
	}
	if (calibParams->reservedUC) {
		free(calibParams->reservedUC);
		calibParams->reservedUC = NULL;
	}
	if (calibParams->reservedV) {
		free(calibParams->reservedV);
		calibParams->reservedV = NULL;
	}
}


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
SCANNER_RESULT allocateCalibrationParameters(CalibrationParameters* calibParams) {
	
	if (calibParams == NULL) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	if (calibParams->C_MatrixSize > 0) {
		calibParams->C1 = (float*)malloc(calibParams->C_MatrixSize * sizeof(float));
		if (!calibParams->C1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		calibParams->C2 = (float*)malloc(calibParams->C_MatrixSize * sizeof(float));
		if (!calibParams->C2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}			
	}

	if (calibParams->RD1_VectorSize > 0) {
		calibParams->RD1 = (float*)malloc(calibParams->RD1_VectorSize * sizeof(float));
		if (!calibParams->RD1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->RD2_VectorSize > 0) {
		calibParams->RD2 = (float*)malloc(calibParams->RD2_VectorSize * sizeof(float));
		if (!calibParams->RD2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->K_MatrixSize > 0) {
		calibParams->K1 = (float*)malloc(calibParams->K_MatrixSize * sizeof(float));
		if (!calibParams->K1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		calibParams->K2 = (float*)malloc(calibParams->K_MatrixSize * sizeof(float));
		if (!calibParams->K2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->RT_MatrixSize > 0) {
		calibParams->RT1 = (float*)malloc(calibParams->RT_MatrixSize * sizeof(float));
		if (!calibParams->RT1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		calibParams->RT2 = (float*)malloc(calibParams->RT_MatrixSize * sizeof(float));
		if (!calibParams->RT2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->reservedISize > 0) {
		calibParams->reservedI = (int*)malloc(calibParams->reservedISize * sizeof(int));
		if (!calibParams->reservedI) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->reservedFSize > 0) {
		calibParams->reservedF = (float*)malloc(calibParams->reservedFSize * sizeof(float));
		if (!calibParams->reservedF) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->reservedUCSize > 0) {
		calibParams->reservedUC = (unsigned char*)malloc(calibParams->reservedUCSize * sizeof(unsigned char));
		if (!calibParams->reservedUC) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	if (calibParams->reservedVSize > 0) {
		calibParams->reservedV = (void*)malloc(calibParams->reservedVSize * calibParams->reservedVBytesPerElement);
		if (!calibParams->reservedV) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}		
	}

	return SCANNER_OK;
}


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
SCANNER_RESULT copyCalibrationParameters(CalibrationParameters* src, CalibrationParameters* trg) {
	if (!src || !trg) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	trg->calibFormat = src->calibFormat;
	trg->reserved1 = src->reserved1;
	trg->reserved2 = src->reserved2;
	trg->C_MatrixSize = src->C_MatrixSize;
	trg->RD1_VectorSize = src->RD1_VectorSize;
	trg->RD2_VectorSize = src->RD2_VectorSize;
	trg->K_MatrixSize = src->K_MatrixSize;
	trg->RT_MatrixSize = src->RT_MatrixSize;
	trg->reservedISize = src->reservedISize;
	trg->reservedFSize = src->reservedFSize;
	trg->reservedUCSize = src->reservedUCSize;
	trg->reservedVSize = src->reservedVSize;
	trg->reservedVBytesPerElement = src->reservedVBytesPerElement;
		
	if (src->C_MatrixSize > 0 && src->C_MatrixSize == trg->C_MatrixSize) {
		memcpy(trg->C1, src->C1, src->C_MatrixSize * sizeof(float));
		memcpy(trg->C2, src->C2, src->C_MatrixSize * sizeof(float));
	}
	if (src->RD1_VectorSize > 0 && src->RD1_VectorSize == trg->RD1_VectorSize) {
		memcpy(trg->RD1, src->RD1, src->RD1_VectorSize * sizeof(float));		
	}
	if (src->RD2_VectorSize > 0 && src->RD2_VectorSize == trg->RD2_VectorSize) {
		memcpy(trg->RD2, src->RD2, src->RD2_VectorSize * sizeof(float));		
	}
	if (src->K_MatrixSize > 0 && src->K_MatrixSize == trg->K_MatrixSize) {
		memcpy(trg->K1, src->K1, src->K_MatrixSize * sizeof(float));
		memcpy(trg->K2, src->K2, src->K_MatrixSize * sizeof(float));
	}
	if (src->RT_MatrixSize > 0 && src->RT_MatrixSize == trg->RT_MatrixSize) {
		memcpy(trg->RT1, src->RT1, src->RT_MatrixSize * sizeof(float));
		memcpy(trg->RT2, src->RT2, src->RT_MatrixSize * sizeof(float));
	}
	if (src->reservedISize > 0 && src->reservedISize == trg->reservedISize) {
		memcpy(trg->reservedI, src->reservedI, src->reservedISize * sizeof(int));		
	}
	if (src->reservedFSize > 0 && src->reservedFSize == trg->reservedFSize) {
		memcpy(trg->reservedF, src->reservedF, src->reservedFSize * sizeof(float));		
	}
	if (src->reservedUCSize > 0 && src->reservedUCSize == trg->reservedUCSize) {
		memcpy(trg->reservedUC, src->reservedUC, src->reservedUCSize * sizeof(unsigned char));		
	}
	int s1 = src->reservedVSize * src->reservedVBytesPerElement;
	int s2 = trg->reservedVSize * trg->reservedVBytesPerElement;
	if (s1 > 0 && s1 == s2) {
		memcpy(trg->reservedV, src->reservedV, s1);		
	}	

	return SCANNER_OK;
}


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
int sizeOfCalibrationParameters(CalibrationParameters* calibParams) {
	if (!calibParams) {
		return 0;
	}
	if (!calibParams->C1 || !calibParams->C2 || !calibParams->RD1 || !calibParams->RD2 || !calibParams->K1 || !calibParams->K2 || !calibParams->RT1 || !calibParams->RT2) {
		return 0;
	}

	int size = 13 * sizeof(int) 
		+ (2*calibParams->C_MatrixSize 
		+ calibParams->RD1_VectorSize
		+ calibParams->RD2_VectorSize
		+ 2*calibParams->K_MatrixSize
		+ 2*calibParams->RT_MatrixSize) * sizeof(float)
		+ calibParams->reservedISize * sizeof(int)
		+ calibParams->reservedFSize * sizeof(float)
		+ calibParams->reservedUCSize * sizeof(unsigned char)
		+ calibParams->reservedVSize * calibParams->reservedVBytesPerElement;
	return size;
}



/***************************************************************************************
* readCalibrationFile()
*
* Description
*	reads calibration file
* Arguments
*	calibrationFileName - file name of calibration results (produced by MATLAB calibration application)
*	calibParams - pointer to CalibrationParameters structure to hold the calibration parameters
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	after using this function, call the clearCalibrationParameters() function to release allocated memory
***************************************************************************************/
SCANNER_RESULT readCalibrationFile(const char* calibrationFileName, CalibrationParameters* calibParams) {
	
	FILE* f;
	errno_t err;			
	SCANNER_RESULT res;

	if (!calibParams) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	err = fopen_s(&f, calibrationFileName, "rb");
	if (err != 0) {
		return SCANNER_CALIBRATION_FILE_NOT_FOUND;
	}	

	res = readCalibrationParameters(f, calibParams);
	if (res != SCANNER_OK) {
		fclose(f);
		return res;
	}	

	fclose(f);
	return SCANNER_OK;
}


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
SCANNER_RESULT readCalibrationParametersFromGipFile(const char* fileName, CalibrationParameters* calibParams, ImageData_GIP* data) {
	
	FILE* f;
	errno_t err;			
	SCANNER_RESULT res;

	if (!calibParams || !data) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	err = fopen_s(&f, fileName, "rb");
	if (err != 0) {
		return SCANNER_CALIBRATION_FILE_NOT_FOUND;
	}	

	int skipSize = data->header.headerSize
					+ (data->header.reservedISize) * sizeof(int)
					+ (data->header.reservedFSize) * sizeof(float)
					+ (data->header.reservedDSize) * sizeof(double)
					+ (data->header.reservedUISize) * sizeof(unsigned int);

	if (fseek(f, skipSize, SEEK_SET)) {
		fclose(f);
		return SCANNER_FILE_READ_FAILURE;
	}

	res = readCalibrationParameters(f, calibParams);
	if (res != SCANNER_OK) {
		fclose(f);
		return res;
	}	

	fclose(f);
	return SCANNER_OK;
}


/***************************************************************************************
* writeCalibrationFile()
*
* Description
*	write calibration file
* Arguments
*	calibrationFileName - file name of calibration results (usually "calibration_CB_Ex.dat")
*	calibParams - pointer to CalibrationParameters structure holding the calibration parameters
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	
***************************************************************************************/
SCANNER_RESULT writeCalibrationFile(const char* calibrationFileName, CalibrationParameters* calibParams) {
	
	FILE* f;
	errno_t err;	
	SCANNER_RESULT res;

	if (!calibParams) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	err = fopen_s(&f, calibrationFileName, "wb");
	if (err != 0) {
		return SCANNER_FILE_OPEN_FAILURE;
	}	

	res = writeCalibrationParameters(f, calibParams);
	if (res != SCANNER_OK) {
		fclose(f);
		return res;
	}	
	
	fclose(f);
	return SCANNER_OK;
}


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
SCANNER_RESULT writeCalibrationParametersToGipFile(const char* fileName, CalibrationParameters* calibParams, ImageData_GIP* data) {
	
	FILE* f;
	errno_t err;	
	SCANNER_RESULT res;

	if (!calibParams || !data) {
		return SCANNER_INVALID_ARGUMENTS;
	}

	err = fopen_s(&f, fileName, "r+b");
	if (err != 0) {
		return SCANNER_FILE_OPEN_FAILURE;
	}	

	int skipSize = data->header.headerSize
					+ (data->header.reservedISize) * sizeof(int)
					+ (data->header.reservedFSize) * sizeof(float)
					+ (data->header.reservedDSize) * sizeof(double)
					+ (data->header.reservedUISize) * sizeof(unsigned int);

	if (fseek(f, skipSize, SEEK_SET)) {
		fclose(f);
		return SCANNER_FILE_WRITE_FAILURE;
	}

	res = writeCalibrationParameters(f, calibParams);
	if (res != SCANNER_OK) {
		fclose(f);
		return res;
	}	
	
	fclose(f);
	return SCANNER_OK;
}


/***************************************************************************************
* readCalibrationParameters
*
* Description
*	Reads calibration parameters from a file into a CalibrationParameters structure
* Arguments
*	f - input file
*	calibParams - pointer to a CalibrationParameters structure in which to store the data
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* 
* Remarks
*	The file should already be opened and will be closed later. 
***************************************************************************************/
SCANNER_RESULT readCalibrationParameters(FILE* f, CalibrationParameters* calibParams) {
	
	if (!fread(&(calibParams->calibFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}

	if (calibParams->calibFormat == 1) {

	}
	else {
		// calibParams->calibFormat value not valid
		SCANNER_FILE_NOT_VALID;
	}

	if (!fread(&(calibParams->reserved1), sizeof(int), 1, f)) {
		fclose(f);
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->reserved2), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->C_MatrixSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->RD1_VectorSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->RD2_VectorSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->K_MatrixSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->RT_MatrixSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->reservedISize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->reservedFSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->reservedUCSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->reservedVSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}
	if (!fread(&(calibParams->reservedVBytesPerElement), sizeof(int), 1, f)) {
		return SCANNER_FILE_READ_FAILURE;
	}	

	if (calibParams->C_MatrixSize > 0) {
		calibParams->C1 = (float*)malloc(calibParams->C_MatrixSize * sizeof(float));
		if (!calibParams->C1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->C1, calibParams->C_MatrixSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}

		calibParams->C2 = (float*)malloc(calibParams->C_MatrixSize * sizeof(float));
		if (!calibParams->C2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->C2, calibParams->C_MatrixSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}	
	}

	if (calibParams->RD1_VectorSize > 0) {
		calibParams->RD1 = (float*)malloc(calibParams->RD1_VectorSize * sizeof(float));
		if (!calibParams->RD1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->RD1, calibParams->RD1_VectorSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->RD2_VectorSize > 0) {
		calibParams->RD2 = (float*)malloc(calibParams->RD2_VectorSize * sizeof(float));
		if (!calibParams->RD2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->RD2, calibParams->RD2_VectorSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->K_MatrixSize > 0) {
		calibParams->K1 = (float*)malloc(calibParams->K_MatrixSize * sizeof(float));
		if (!calibParams->K1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->K1, calibParams->K_MatrixSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}

		calibParams->K2 = (float*)malloc(calibParams->K_MatrixSize * sizeof(float));
		if (!calibParams->K2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->K2, calibParams->K_MatrixSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->RT_MatrixSize > 0) {
		calibParams->RT1 = (float*)malloc(calibParams->RT_MatrixSize * sizeof(float));
		if (!calibParams->RT1) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->RT1, calibParams->RT_MatrixSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}

		calibParams->RT2 = (float*)malloc(calibParams->RT_MatrixSize * sizeof(float));
		if (!calibParams->RT2) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->RT2, calibParams->RT_MatrixSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->reservedISize > 0) {
		calibParams->reservedI = (int*)malloc(calibParams->reservedISize * sizeof(int));
		if (!calibParams->reservedI) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->reservedI, calibParams->reservedISize * sizeof(int), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->reservedFSize > 0) {
		calibParams->reservedF = (float*)malloc(calibParams->reservedFSize * sizeof(float));
		if (!calibParams->reservedF) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->reservedF, calibParams->reservedFSize * sizeof(float), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->reservedUCSize > 0) {
		calibParams->reservedUC = (unsigned char*)malloc(calibParams->reservedUCSize * sizeof(unsigned char));
		if (!calibParams->reservedUC) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->reservedUC, calibParams->reservedUCSize * sizeof(unsigned char), 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	if (calibParams->reservedVSize > 0) {
		calibParams->reservedV = (void*)malloc(calibParams->reservedVSize * calibParams->reservedVBytesPerElement);
		if (!calibParams->reservedV) {
			clearCalibrationParameters(calibParams);
			return SCANNER_MEMORY_ALLOCATION_FAILURE;
		}
		if (!fread(calibParams->reservedV, calibParams->reservedVSize * calibParams->reservedVBytesPerElement, 1, f)) {
			clearCalibrationParameters(calibParams);
			return SCANNER_FILE_READ_FAILURE;
		}
	}

	return SCANNER_OK;
}


/***************************************************************************************
* writeCalibrationParameters
*
* Description
*	Write calibration parameters into a file.
* Arguments
*	f - output file
*	calibParams - pointer to a CalibrationParameters structure containing the data
* Return value
*	SCANNER_OK if successful, failure indications otherwise.
* Remarks
*	The file should already be opened and will be closed later.
***************************************************************************************/
SCANNER_RESULT writeCalibrationParameters(FILE* f, CalibrationParameters* calibParams) {
	
	if (!fwrite(&(calibParams->calibFormat), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}	
	if (!fwrite(&(calibParams->reserved1), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->reserved2), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->C_MatrixSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->RD1_VectorSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->RD2_VectorSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->K_MatrixSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->RT_MatrixSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->reservedISize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->reservedFSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->reservedUCSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->reservedVSize), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}
	if (!fwrite(&(calibParams->reservedVBytesPerElement), sizeof(int), 1, f)) {
		return SCANNER_FILE_WRITE_FAILURE;
	}	

	if (calibParams->C_MatrixSize > 0) {
		if (!fwrite(calibParams->C1, calibParams->C_MatrixSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
		if (!fwrite(calibParams->C2, calibParams->C_MatrixSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}	
	}

	if (calibParams->RD1_VectorSize > 0) {
		if (!fwrite(calibParams->RD1, calibParams->RD1_VectorSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->RD2_VectorSize > 0) {
		if (!fwrite(calibParams->RD2, calibParams->RD2_VectorSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->K_MatrixSize > 0) {
		if (!fwrite(calibParams->K1, calibParams->K_MatrixSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
		if (!fwrite(calibParams->K2, calibParams->K_MatrixSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->RT_MatrixSize > 0) {
		if (!fwrite(calibParams->RT1, calibParams->RT_MatrixSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
		if (!fwrite(calibParams->RT2, calibParams->RT_MatrixSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->reservedISize > 0) {
		if (!fwrite(calibParams->reservedI, calibParams->reservedISize * sizeof(int), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->reservedFSize > 0) {
		if (!fwrite(calibParams->reservedF, calibParams->reservedFSize * sizeof(float), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->reservedUCSize > 0) {
		if (!fwrite(calibParams->reservedUC, calibParams->reservedUCSize * sizeof(unsigned char), 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	if (calibParams->reservedVSize > 0) {
		if (!fwrite(calibParams->reservedV, calibParams->reservedVSize * calibParams->reservedVBytesPerElement, 1, f)) {
			return SCANNER_FILE_WRITE_FAILURE;
		}
	}

	return SCANNER_OK;
}