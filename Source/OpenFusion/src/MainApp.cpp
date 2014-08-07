/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
Volume rendering sample

This sample loads a 3D volume from disk and displays it using
ray marching and 3D textures.

Note - this is intended to be an example of using 3D textures
in CUDA, not an optimized volume renderer.

Changes
sgg 22/3/2010
- updated to use texture for display instead of glDrawPixels.
- changed to render from front-to-back rather than back-to-front.
*/

// OpenGL Graphics includes
#include "GL/glew.h"
#if defined (__APPLE__) || defined(MACOSX)
#include "GLUT/glut.h"
#else
#include "GL/freeglut.h"
#endif

// CUDA utilities and system includes
#include "cutil_inline.h"    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h

// CUDA Includes
#include <vector_functions.h>
#include <driver_functions.h>

#include "MainApp.h"
#include "FileSystemHelper.h"
#include "utilities.h"
#include "General.h"

#include "CImg.h"

#include <iostream>
#include <fstream> // TODO REMOVE


using namespace cimg_library;
using namespace std;


dim3 gridSize;

unsigned int timer = 0;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling


/*************** GLUT Callbacks *****************/
void displayCallBack();
void motionCallBack(int x, int y);
	void mouseCallBack(int button, int state, int x, int y);
void reshapeCallBack(int w, int h);
void keyboardCallBack(unsigned char key, int x, int y);
void idleCallBack();

void initPixelBuffer();
void cleanup();

void computeFPS()
{
	fpsCount++;
	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
		sprintf(fps, "OpenFusion | %3.1f fps", ifps);
		glutSetWindowTitle(fps);
		fpsCount = 0;
		cutilCheckError(cutResetTimer(timer));  
	}
}

bool isInsideVolume(float* volume_host, int x, int y, int z) {
	if (x == VOLUME_SIZE || y == VOLUME_SIZE || z == VOLUME_SIZE) {
		return false;
	}

	return volume_host[x*VOLUME_SIZE*VOLUME_SIZE + y*VOLUME_SIZE + z] < 0;
}

void VerticalFlip(CImg<unsigned char>* img)
{
    const int width = img->width();
    const int height = img->height();
    const int spectrum = img->spectrum();

    int temp;
    for(int x=0; x<width; x++)
    {
        for (int y=0; y<height/2; y++)
        {
            for (int z=0; z<spectrum; z++)
            {
				temp = *(img->data(x,y,0,z));
				*(img->data(x,y,0,z)) = *(img->data(x,height-1-y,0,z));
				*(img->data(x,height-1-y,0,z)) = temp;
            }
        }
    }
}

void RunApp::saveImageToFile() {	
	const int buffer = PBO_FILE;
	glBindTexture(GL_TEXTURE_2D, pboArr[buffer].tex);

	uint* textureBuffer = new uint[m_width*m_height];
	if (!textureBuffer) {
		cout << "Out of memory." << endl;
		exit(1);
	}

	// TODO CHANGE FROM RGBA -> RGB
	glCopyTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, 0, m_height, m_width, m_height, 0 ); // Copies the top left quarter of the screen
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureBuffer);

	unsigned char* textureBufferSerialized = new unsigned char[m_width*m_height*3];
	if (!textureBufferSerialized) {
		cout << "Out of memory." << endl;
		delete[] textureBuffer;
		exit(1);
	}

	// Convert buffer to CImg format: Instead of RGBRGB...RGB, the new format is R(0,0) R(0,1)...R(WIDTH,HEIGHT) G(0,0) G(0,1) etc.
	for (int i = 0; i < m_width; ++i) {
		for (int j = 0; j < m_height; ++j) {
			const uint rgba = textureBuffer[i*m_height + j];
			for (int k = 0; k < 3; ++k) {
				textureBufferSerialized[k*m_width*m_height + i*m_height + j] = ((const char*)(&rgba))[k];
			}
		}
	}

	CImg<unsigned char> textureBufferMatrix(textureBufferSerialized, m_width, m_height, 1, 3, false);
	VerticalFlip(&textureBufferMatrix);

	// The filename is the e-mail address of the model.
	if (savesCounter == 1) {
		cout << endl << "Please write your e-mail address: ";
		scanf("%s", emailAddress);
	}

	// Create a folder named "Screenshots" if doesn't exist.
	FileHandler* fileHandler = FileHandler::Init();
	const char* dirName = "Screenshots";
	fileHandler->CreateDir(dirName);

	char fileName[EMAIL_LENGTH + 4];
	sprintf(fileName, "%s.bmp", emailAddress);

	char fullAddress[EMAIL_LENGTH + 25];
	fileHandler->GetPath(dirName, fileName, fullAddress);

	textureBufferMatrix.save(fullAddress, savesCounter); // savesCounter is the suffix of the filename.
	cout << "Successfully wrote file to disk." << endl;
	savesCounter++;

	delete[] textureBuffer;
	delete[] textureBufferSerialized;
	delete fileHandler;
}


void RunApp::saveVolumeToFile() {

	// TODO: FIGURE HOW TO INTERACTIVELY SAVE A FILE TO THE DISK (THE USER GIVES THE NAME OF THE FILE)


	float* volume_host_float = new float[VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE];
	if (!volume_host_float) {
		cout << "Failed to allocate memory for volume on host." << endl;
		return;
	}

	cutilSafeCall(cudaMemcpy(volume_host_float, d_voxel_array, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(float), cudaMemcpyDeviceToHost));

	//float* volume_host_for_MC = new float[VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE];
	//if (!volume_host_for_MC) {
	//	cout << "Failed to allocate memory for volume on host." << endl;
	//	delete[] volume_host_float;
	//	return;
	//}

	/*signed char* volume_host_char = new signed char[VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE];
	if (!volume_host_char) {
		cout << "Failed to allocate memory for volume on host." << endl;
		return;
	}

	for (long i = 0; i < VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE; ++i) {
		float res = volume_host_float[i];
		if (res > 127 || res < -127) {
			cout << "Overflow: " << res << endl;
		}
		volume_host_char[i] = res;
	}*/

	//     3-----2
	//    /|    /|
	//    7----6 |
	//    ||   | |
	//    |0---|-1
	//    4----5/

	// TODO: ARE YOU INSANE???? IT'S UNSIGNED CHAR!!!!!! IT WON'T WORK.
	/*
	for (int x = 0; x < VOLUME_SIZE; ++x) {
		for (int y = 0; y < VOLUME_SIZE; ++y) {
			for (int z = 0; z < VOLUME_SIZE; ++z) {
				uchar result = 0;
				bool point[8] = { 0 };
				point[0] = isInsideVolume(volume_host_float, x, y, z);
				point[1] = isInsideVolume(volume_host_float, x+1, y, z);
				point[2] = isInsideVolume(volume_host_float, x+1, y+1, z);
				point[3] = isInsideVolume(volume_host_float, x, y+1, z);
				point[4] = isInsideVolume(volume_host_float, x, y, z+1);
				point[5] = isInsideVolume(volume_host_float, x+1, y, z+1);
				point[6] = isInsideVolume(volume_host_float, x+1, y+1, z+1);
				point[7] = isInsideVolume(volume_host_float, x, y+1, z+1);

				for (int i = 0; i < 8; ++i) {	// TODO FOR CHAR
					result |= point[i] << i;	// Or between the current result and the shifted boolean
				}

				//for (int i = 0; i < 8; ++i) {	// TODO FOR FLOAT
				//	result += ( (point[i] ? 1 : 0) * pow(2.0, i));	// Or between the current result and the shifted boolean
				//}

				if (_isnan(result)) {
					cout << "NAN!!!" << endl;
				}

				volume_host_char[x*VOLUME_SIZE*VOLUME_SIZE + y*VOLUME_SIZE + z] = result;

				//volume_host_for_MC[x*VOLUME_SIZE*VOLUME_SIZE + y*VOLUME_SIZE + z] = result;
			}
		}
	}*/
	
	

	char fileName[] = "SavedVolume.raw";	// TODO: MAKE IT GENERAL BY INPUT IN CONSOLE
	ofstream outfile(fileName, ofstream::binary);
	outfile.clear();

	// write to outfile
	//outfile.write ((const char*)volume_host_for_MC, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(float));
	//outfile.write ((const char*)volume_host_char, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(char));
	outfile.write ((const char*)volume_host_float, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(float));
	outfile.flush();
	outfile.close();

	// maybe could be deleted earlier
	delete[] volume_host_float;

	cout << "Writing volume to disk is done." << endl;

	return;

	/*
	float* volume_MC = NULL;
	cutilSafeCall(cudaMalloc((void**)&volume_MC, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(float))); // TODO - MAYBE CHANGE THIS TO CHAR TO SAVE SPACE
	cutilSafeCall(cudaMemcpy(volume_MC, volume_host_for_MC, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(float), cudaMemcpyHostToDevice));

	delete[] volume_host_for_MC;
	
	cleanup();

	CUDAMarchingCubes MC;
	uint3 MCgridSize = make_uint3(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE);

	float3* triangles = NULL;
	int trianglesLength = 600000;
	cutilSafeCall(cudaMalloc((void**)&triangles, trianglesLength*sizeof(float3)));

	if (!MC.computeIsosurface(volume_MC, NULL, MC.RGB3F, MCgridSize, make_float3(VOLUMEMINX, VOLUMEMINY, VOLUMEMINZ),
							  make_float3(worldVolume, worldVolume, worldVolume), true, triangles, NULL, NULL, trianglesLength)) {
		cout << "COMPUTE FAILED" << endl;
	}

	float3* triangles_host = new float3[trianglesLength];
	cutilSafeCall(cudaMemcpy(triangles_host, triangles, trianglesLength*sizeof(float3), cudaMemcpyDeviceToHost));

	char fileName2[] = "triangles.obj";
	ofstream marchOut(fileName2, ofstream::binary);
	marchOut.clear();

	// write to outfile
	//marchOut.write ((const char*)triangles_host, trianglesLength*sizeof(float3));

	marchOut << "# List of vertices" << endl;
	for (int i = 0; i < trianglesLength; ++i) {
		marchOut << "v " << triangles_host[i].x << " " << triangles_host[i].y << " " << triangles_host[i].z << endl;
	}
	marchOut.flush();

	marchOut << endl << "# List of faces" << endl;
	for (int i = 0; i < trianglesLength; i += 3) {
		marchOut << "f " << i << " " << i+1 << " " << i + 2 << endl;
	}
	marchOut << endl;

	marchOut.flush();
	marchOut.close();
	
	cout << "DONE MC" << endl;

	cutilSafeCall(cudaFree(triangles));
	cutilSafeCall(cudaFree(volume_MC));
	delete[] volume_host_for_MC;
	*/
}


// render image using CUDA
void RunApp::render()
{
	//copyInvViewMatrix(invViewMatrix, sizeof(float4)*3); 

	// TODO - we assume that the inverse view matrix is copied into the render's constant variables during initialization
	// and between each call to render.

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	cutilSafeCall(cudaGraphicsMapResources(1, &pboArr[PBO_MAIN].cuda_pbo_resource, 0));
	size_t num_bytes; 
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, pboArr[PBO_MAIN].cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	cutilSafeCall(cudaMemset(d_output, 0, m_width*m_height*sizeof(uint))); // sizeof(uint) == 4

	// call CUDA kernel, writing results to PBO 
	m_raycaster->Raycast(d_output, d_currentTransform, d_voxel_array, d_renderedVertexMap, d_cameraPosition, showNormalsFlag);

	cutilCheckMsg("kernel failed");

	cutilSafeCall(cudaGraphicsUnmapResources(1, &pboArr[PBO_MAIN].cuda_pbo_resource, 0));

}

void RunApp::drawTextureToScreen(int buffer)
{
	// copy from pbo to texture

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pboArr[buffer].pbo);
	glBindTexture(GL_TEXTURE_2D, pboArr[buffer].tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, buffer);

	// draw textured quad
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	
	switch (buffer)
	{
	case PBO_MAIN:				glTexCoord2f(0, 0); glVertex2f(0, 1);		// THE MAIN OPENGL IS NOW UPSIDE DOWN TOO!!!!!
								glTexCoord2f(1, 0); glVertex2f(0.5, 1);
								glTexCoord2f(1, 1); glVertex2f(0.5, 0.5);
								glTexCoord2f(0, 1); glVertex2f(0, 0.5);
								break;

	case PBO_DEPTH:				glTexCoord2f(0, 0); glVertex2f(0, 0.5);		// PAY ATTENTION TO THIS TRICK TO TURN THIS IMAGE UPSIDE DOWN
								glTexCoord2f(1, 0); glVertex2f(0.5, 0.5);
								glTexCoord2f(1, 1); glVertex2f(0.5, 0);
								glTexCoord2f(0, 1); glVertex2f(0, 0);
								break;

	case PBO_NEW_NORMAL:		glTexCoord2f(0, 0); glVertex2f(0.5, 1);		// PAY ATTENTION TO THIS TRICK TO TURN THIS IMAGE UPSIDE DOWN
								glTexCoord2f(1, 0); glVertex2f(1, 1);
								glTexCoord2f(1, 1); glVertex2f(1, 0.5);
								glTexCoord2f(0, 1); glVertex2f(0.5, 0.5);
								break;

	case PBO_RGB:				glTexCoord2f(0, 0); glVertex2f(0.5, 0.5);		// PAY ATTENTION TO THIS TRICK TO TURN THIS IMAGE UPSIDE DOWN
								glTexCoord2f(1, 0); glVertex2f(1, 0.5);
								glTexCoord2f(1, 1); glVertex2f(1, 0);
								glTexCoord2f(0, 1); glVertex2f(0.5, 0);
								break;
	}


	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, buffer);
}



void RunApp::idle()
{
	glutPostRedisplay();
}

void RunApp::keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case 27:	// Escape
		break;
	case 'f':
		if(!writingModeOn && !startTracking){
			cameraHandler->SwitchToReadMode();
		}
		break;
	case 'i':
		if (!writingModeOn) {
			startTracking = true;
		}
		break;
	case 't':
		if(!tPressed && !writingModeOn){
			terminateTracking = true;
			cutilSafeCall(cudaMemcpy(d_savedTransform, d_currentTransform, 4*4*sizeof(float), cudaMemcpyDeviceToDevice));
			tPressed = true;
		}
		break;
	case 'T':
		if(tPressed && !writingModeOn)
		{
			terminateTracking = false;
			cutilSafeCall(cudaMemcpy(d_currentTransform, d_savedTransform, 4*4*sizeof(float), cudaMemcpyDeviceToDevice));
			General::Instance()->UpdateTransforms(d_currentTransform,
						d_invCurrentTransform,
						d_cameraPosition,
						NULL);
			cudaDeviceSynchronize();
			//m_raycaster->CopyCameraPosition(d_cameraPosition);
			cudaDeviceSynchronize();
			tPressed = false;
		}
		break;
	case 'N':
		showNormalsFlag = true;
		break;
	case 'n': showNormalsFlag = false;
		break;

	case 's':
		if (startTracking && !writingModeOn) {
			tPressed = true;
			terminateTracking = true;
			saveImageToFile();
		}
		break;
	case 'r':
		glutLeaveMainLoop();
		break;
	case 'v':
		if (startTracking && !writingModeOn) {
			tPressed = true;
			terminateTracking = true;
			saveVolumeToFile();
		}
		break;
	case 'w':
		if (!startTracking && !writingModeOn){
			writingModeOn = true;
			cameraHandler->SwitchToWriteMode();
			// TODO FIGURE OUT A WAY FOR FOCUSING THE GLUT WINDOW AGAIN
		}
		break;
	case 'W':
		if (writingModeOn){
			glutLeaveMainLoop();
		}
		break;
	default:
		break;
	}

	glutPostRedisplay();
}


void RunApp::cleanup()
{
	cutilSafeCall(cudaDeviceSynchronize());

	cutilCheckError( cutDeleteTimer( timer));

	for(int i = 0; i < NUM_OF_PBO; i++)
	{
		if (pboArr[i].pbo)
		{
			//cudaGraphicsUnregisterResource(pboArr[i].cuda_pbo_resource); // No need to unregister, because we unregister every frame
			glDeleteBuffersARB(1, &pboArr[i].pbo);
			glDeleteTextures(1, &pboArr[i].tex);
		}
	}

	///****************OUR STUFF****************-/
	delete icp_module;
	delete cameraHandler;
	cudaFree(d_renderedVertexMap.vertex);
	cudaFree(d_renderedVertexMap.normal);
	cudaFree(d_voxel_array);
	cudaFree(d_weight_array);
	cudaFree(d_cameraPosition);
	cudaFree(d_currentTransform);
	cudaFree(d_savedTransform);
	cudaFree(d_invCurrentTransform);
	cudaFree(d_newDepthMap);
	cudaFree(newDepthData.depth);
	cudaFree(newDepthData.vertex);
	cudaFree(newDepthData.normal);

	///*****************************************-/
}

void RunApp::initGL(int *argc, char **argv)
{
	// initialize GLUT callback functions
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(m_width*2, m_height*2);			// TODO - PAY ATTENTION TO THIS TRICKY 2
	glutCreateWindow("CUDA volume rendering");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION); // TODO - THIS IS NEW. TO CONTINUE EXECUTING AFTER EXITING THE MAIN LOOP

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
		cout << "Error: glew is not supported" <<endl;
	}
}

void RunApp::initPixelBuffer()
{
	for(int i = 0; i < NUM_OF_PBO; ++i)
	{
		if (pboArr[i].pbo) {
			// unregister this buffer object from CUDA C
			cutilSafeCall(cudaGraphicsUnregisterResource(pboArr[i].cuda_pbo_resource));

			// delete old buffer
			glDeleteBuffersARB(1, &pboArr[i].pbo);
			glDeleteTextures(1, &pboArr[i].tex);
		}

		// create pixel buffer object for display
		glGenBuffersARB(1, &pboArr[i].pbo);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pboArr[i].pbo);
		glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, m_width*m_height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, i);

		// register this buffer object with CUDA
		cutilSafeCall(cudaGraphicsGLRegisterBuffer(&pboArr[i].cuda_pbo_resource, pboArr[i].pbo, cudaGraphicsMapFlagsWriteDiscard));

		// create texture for display
		glGenTextures(1, &pboArr[i].tex);
		glBindTexture(GL_TEXTURE_2D, pboArr[i].tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, i);
	}

}

// General initialization call for CUDA Device
int RunApp::chooseCudaDevice(int argc, char **argv, bool bUseOpenGL)
{
	int result = 0;
	if (bUseOpenGL) {
		result = cutilChooseCudaGLDevice();
	} else {
		result = cutilChooseCudaDevice();
	}
	return result;
}

void RunApp::initCudaPointers()
{
	//m_raycaster->CopyKInvMatrix(KinvMat);	// copy kinvMat to cuda in volumerRender_kernel

	// Initialize voxel volume
	cutilSafeCall(cudaMalloc((void**)&d_voxel_array, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(VoxelType)));
	cutilSafeCall(cudaMalloc((void**)&d_weight_array, VOLUME_SIZE*VOLUME_SIZE*VOLUME_SIZE*sizeof(VoxelWeightType)));
	General::Instance()->InitVoxelArray(d_voxel_array, d_weight_array);
	
	// Allocate rendered vertex map - vertex
	cutilSafeCall(cudaMalloc((void**)&(d_renderedVertexMap.vertex), m_width*m_height*sizeof(float3)));
	cutilSafeCall(cudaMemset(d_renderedVertexMap.vertex, 0, m_width*m_height*sizeof(float3)));

	// Allocate rendered vertex map - normal
	cutilSafeCall(cudaMalloc((void**)&(d_renderedVertexMap.normal), m_width*m_height*sizeof(float3)));
	cutilSafeCall(cudaMemset(d_renderedVertexMap.normal, 0, m_width*m_height*sizeof(float3)));

	// Allocate new depth map
	cutilSafeCall(cudaMalloc((void**)&d_newDepthMap, m_width*m_height*sizeof(CAMERA_DEPTH_TYPE)));
	cutilSafeCall(cudaMemset(d_newDepthMap, 0, m_width*m_height*sizeof(CAMERA_DEPTH_TYPE)));

	// Initialize the currentTransform matrix
	float MView[16] = { 1,	0,	0,	0,
						0,	1,	0,	0,
						0,	0,	1,	CAMERA_START_Z_MM,
						0,	0,	0,	1};
	cutilSafeCall(cudaMalloc((void**)&d_currentTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMemcpy(d_currentTransform, MView, 4*4*sizeof(float), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc((void**)&d_savedTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMemcpy(d_savedTransform, d_currentTransform, 4*4*sizeof(float), cudaMemcpyDeviceToDevice));

	// Initialize inverse transformation matrix 
	float InvMView[16] = {	1,	0,	0,	0,
							0,	1,	0,	0,
							0,	0,	1,	-CAMERA_START_Z_MM,
							0,	0,	0,	1};
	cutilSafeCall(cudaMalloc((void**)&d_invCurrentTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMemcpy(d_invCurrentTransform, InvMView, 4*4*sizeof(float), cudaMemcpyHostToDevice));

	// Initialize camera position
	float3 cameraPosition = make_float3(0.f, 0.f, CAMERA_START_Z_MM);
	cutilSafeCall(cudaMalloc((void**)&d_cameraPosition, sizeof(float3)));
	cutilSafeCall(cudaMemcpy(d_cameraPosition, &cameraPosition, sizeof(float3), cudaMemcpyHostToDevice));
	//m_raycaster->CopyCameraPosition(d_cameraPosition); //set camera position as constant global variable in volumeRender_kernel.

	//initialize new depth data - depth
	cutilSafeCall(cudaMalloc((void**)&(newDepthData.depth), m_width*m_height*sizeof(CAMERA_DEPTH_TYPE)));
	cutilSafeCall(cudaMemset(newDepthData.depth, 0, m_width*m_height*sizeof(CAMERA_DEPTH_TYPE)));

	//initialize new depth data - vertex
	cutilSafeCall(cudaMalloc((void**)&(newDepthData.vertex), m_width*m_height*sizeof(float3)));
	cutilSafeCall(cudaMemset(newDepthData.vertex, 0, m_width*m_height*sizeof(float3)));

	//initialize new depth data - normal
	cutilSafeCall(cudaMalloc((void**)&(newDepthData.normal), m_width*m_height*sizeof(float3)));
	cutilSafeCall(cudaMemset(newDepthData.normal, 0, m_width*m_height*sizeof(float3)));
}

RunApp* RunApp::thisClass;

RunApp::RunApp(int argc, char** argv) : writingModeOn(false) {

	thisClass = this;
	viewRotation = make_float3(0.f, 0.f, 0.f);
	viewTranslation = make_float3(0.f, 0.f, 0.f);
	viewTranslation = make_float3(0.0, 0.0, -CAMERA_START_Z_OPENGL); //TODO - where do we want the camera to start??? // this was -4.0f
	ox = 0;
	oy = 0;
	buttonState = 0;
		
	d_voxel_array = NULL;
	d_weight_array = NULL;

	icp_module = NULL;
	m_volumeIntegrator = NULL;
	cameraHandler = NULL;
	d_cameraPosition = NULL;

	d_renderedVertexMap.vertex = NULL;
	d_renderedVertexMap.normal = NULL;

	d_currentTransform = NULL;
	d_invCurrentTransform = NULL;
	d_newDepthMap = NULL;

	d_savedTransform = NULL;
	newDepthData.depth = NULL;
	newDepthData.normal = NULL;
	newDepthData.vertex = NULL;

	initDone = false;
	startTracking = false;
	terminateTracking = false;
	showNormalsFlag = false;
	tPressed = false;
	iteration = 1;

	memset(emailAddress, 0, EMAIL_LENGTH);
	savesCounter = 1;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	chooseCudaDevice(argc, argv, true);
	cutilSafeCall(cudaDeviceReset()); // Reset cuda device, release any previously held memory.
	cutilSafeCall(cudaDeviceSynchronize());

	//Initialize camera:
	cameraHandler = new CameraHandler(); // TODO WHILE ASUS CAMRERA IS NOT CONNECTED
	m_width = cameraHandler->GetCameraParams()->m_width;
	m_height = cameraHandler->GetCameraParams()->m_height;
	m_raycaster = new Raycaster(cameraHandler->GetCameraParams());

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL( &argc, argv );


	// calculate new grid size
	gridSize = dim3(iDivUp(m_width, blockSize.x), iDivUp(m_height, blockSize.y));

	///////////////ours/////////////////
	cutilSafeCall(cudaDeviceSynchronize());
	initCudaPointers();
	cutilSafeCall(cudaDeviceSynchronize());

	// Allocate modules
	icp_module = new ICP(gridSize, blockSize, cameraHandler->GetCameraParams());
	m_volumeIntegrator = new VolumeIntegrator(cameraHandler->GetCameraParams());
	
	// TODO THE NEXT SECTION IS POINT CLOUD PROCESSING AND TSDF INTEGRATION
	cameraHandler->cameraIteration(NULL, NULL, NULL, newDepthData, d_currentTransform, NULL, gridSize, blockSize);
	cutilSafeCall(cudaDeviceSynchronize());
	cameraHandler->cameraIteration(d_newDepthMap, NULL, NULL, newDepthData, d_currentTransform, NULL, gridSize, blockSize);
	cutilSafeCall(cudaDeviceSynchronize());
	
	PrintWelcomeMessage();
	
	//end of ours -------------------------------------------------------------------------------------------------------//
	
	cutilCheckError( cutCreateTimer( &timer ) );

	MainLoop();
}

RunApp::~RunApp() {
	cleanup();
	cutilSafeCall(cudaDeviceReset());
	cout << "Exiting..." << endl;
}

void RunApp::PrintWelcomeMessage() {
	cout << "Initialization done." << endl;
	cout << "Program Controls:" << endl;
	cout << "Press 'i' to start tracking" << endl;
	cout << "Press 't' to explore volume" << endl;
	cout << "Press 's' to save current view" << endl;
	cout << "Press 'w' to write depth stream to files" << endl;
	cout << "Press 'W' to stop writing depth stream to files" << endl;
	cout << "Press 'v' to save volumetric model to file" << endl;
	cout << "Press 'N' to view surface normals and 'n' for phong shading" << endl;
	cout << "Press 'r' to reset scan" << endl;
	cout << "Press 'f' to read from file" << endl;
	cout << "Enjoy!" << endl << endl;
}

void RunApp::MainLoop() {
	// This is the normal rendering path for VolumeRender
	glutKeyboardFunc(keyboardCallBack);
	glutMouseFunc(mouseCallBack);
	glutMotionFunc(motionCallBack);
	glutIdleFunc(idleCallBack);
	glutReshapeFunc(reshapeCallBack);
	glutDisplayFunc(displayCallBack);

	initPixelBuffer();
	
	//atexit(cleanup);
	
	initDone = true;

	glutMainLoop();
}

// display results using OpenGL (called by GLUT)
void RunApp::display()
{
	if (initDone) {
		if (!terminateTracking && startTracking) {
			if (iteration % 100 == 0) {
				cout << "Iteration #" << iteration << endl;
			}
			iteration++;
		}

		cutilCheckError(cutStartTimer(timer));

		// Volume integration
		if (!terminateTracking && startTracking) {
			cudaDeviceSynchronize();
			m_volumeIntegrator->Integrate(d_voxel_array, d_weight_array, d_invCurrentTransform, newDepthData, d_cameraPosition);
		}

		// use OpenGL to build view matrix
		GLfloat modelView[16];
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
		glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
		glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
		glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
		glPopMatrix();

		// VERED: OPENGL USES Rt AND ALSO TRANSLATION IN THE LAST ROW (AND NOT LAST COLUMN)
		// THAT'S WHY WE HAVE TO "INVERSE" THIS MATRIX, BUT THIS IS ESSENTIALLY JUST A TRANSPOSE.
		// INVVIEWMATRIX HOLDS THE "STRAIGHT" MODELVIEW MATRIX AS WE KNOW IT.
		invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
		invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
		invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

		cutilSafeCall(cudaDeviceSynchronize());
		render();

		// TODO: NOTE THAT THERE'S NO NEED TO SYNCHRONIZE AFTER THIS KERNEL WHEN CALLING OPENGL FUNCTIONS.
		// THE CUDA-GRAPHICS-UNMAP-RESOURCES GUARANTEES THIS SYNCHRONIZATION.
		// WHEN WE WANT CUDA TO ACCESS THIS MEMORY WE HAVE TO SYNCHRONIZE (HAPPENS LATER)

		// display results
		glClear(GL_COLOR_BUFFER_BIT);

		// draw image from PBO
		glDisable(GL_DEPTH_TEST);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for(int i = 0; i < NUM_OF_PBO; ++i)
		{
			if (i == PBO_FILE) {
				continue;
			}
			drawTextureToScreen(i);
		}
		glutSwapBuffers();

		/*
		// DEBUGGING OF RENDERED VERTEX MAP AND NORMALS
		if (iteration >= 500 && iteration <= 620) {
			
			char filename_vertex[150] = {0};
			sprintf(filename_vertex, "debugging-rendered-vertex\\renderedvertex%d", iteration);
			ofstream outfile_vertex(filename_vertex, ofstream::binary);

			float3* renderedVertex_host = new float3[m_width*m_height];
			cutilSafeCall( cudaMemcpy(renderedVertex_host, d_renderedVertexMap.vertex, m_width*m_height*sizeof(float3), cudaMemcpyDeviceToHost) );

			outfile_vertex.clear();

			// write to outfile
			outfile_vertex.write ((const char*)renderedVertex_host, m_width*m_height*sizeof(float3));
			outfile_vertex.flush();
			outfile_vertex.close();

			char filename_normal[150] = {0};
			sprintf(filename_normal, "debugging-rendered-normal\\renderednormal%d", iteration);
			ofstream outfile_normal(filename_normal, ofstream::binary);

			float3* renderedNormal_host = new float3[m_width*m_height];
			cutilSafeCall( cudaMemcpy(renderedNormal_host, d_renderedVertexMap.normal, m_width*m_height*sizeof(float3), cudaMemcpyDeviceToHost) );

			outfile_normal.clear();

			// write to outfile
			outfile_normal.write ((const char*)renderedNormal_host, m_width*m_height*sizeof(float3));
			outfile_normal.flush();
			outfile_normal.close();

			delete[] renderedVertex_host;
			delete[] renderedNormal_host;
		}

		
		// DEBUGGING OF DEPTH VERTEX MAP AND NORMALS
		if (iteration >= 500 && iteration <= 620) {
			
			char filename_vertex[150] = {0};
			sprintf(filename_vertex, "debugging-new_depth-vertex\\renderedvertex%d", iteration);
			ofstream outfile_vertex(filename_vertex, ofstream::binary);

			float3* depthMapVertex_host = new float3[m_width*m_height];
			cutilSafeCall( cudaMemcpy(depthMapVertex_host, newDepthData.vertex, m_width*m_height*sizeof(float3), cudaMemcpyDeviceToHost) );

			outfile_vertex.clear();

			// write to outfile
			outfile_vertex.write ((const char*)depthMapVertex_host, m_width*m_height*sizeof(float3));
			outfile_vertex.flush();
			outfile_vertex.close();

			char filename_normal[150] = {0};
			sprintf(filename_normal, "debugging-new_depth-normal\\renderednormal%d", iteration);
			ofstream outfile_normal(filename_normal, ofstream::binary);

			float3* depthMapNormal_host = new float3[m_width*m_height];
			cutilSafeCall( cudaMemcpy(depthMapNormal_host, newDepthData.normal, m_width*m_height*sizeof(float3), cudaMemcpyDeviceToHost) );

			outfile_normal.clear();

			// write to outfile
			outfile_normal.write ((const char*)depthMapNormal_host, m_width*m_height*sizeof(float3));
			outfile_normal.flush();
			outfile_normal.close();

			delete[] depthMapVertex_host;
			delete[] depthMapNormal_host;
		}
		*/

		//-----------------our part----------------------------------------------------------------------

		// Core functions of OpenFusion

		// TODO: CAMERAINPUT DOESN'T USE RENDER RESULTS, SO NO NEED TO SYNCHRONIZE BEFORE IT

		if (!terminateTracking) {
			uint *newNormalMap;
			CAMERA_RGB_TYPE* newRgbMap;
			uint *newDepthMap;
			// map PBO to get CUDA device pointer

			cutilSafeCall(cudaGraphicsMapResources(1, &pboArr[PBO_NEW_NORMAL].cuda_pbo_resource, 0));
			cutilSafeCall(cudaGraphicsMapResources(1, &pboArr[PBO_RGB].cuda_pbo_resource, 0));
			cutilSafeCall(cudaGraphicsMapResources(1, &pboArr[PBO_DEPTH].cuda_pbo_resource, 0));
			size_t num_bytes;
			cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&newNormalMap, &num_bytes, pboArr[PBO_NEW_NORMAL].cuda_pbo_resource));
			cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&newRgbMap, &num_bytes, pboArr[PBO_RGB].cuda_pbo_resource));
			cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&newDepthMap, &num_bytes, pboArr[PBO_DEPTH].cuda_pbo_resource));
			//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

			// clear image
			cutilSafeCall(cudaMemset(newNormalMap, 0, m_width*m_height*sizeof(uint))); // sizeof(uint) == 4 // TODO MAYBE REDUNDANT. WE WRITE ON ALL IMAGE
			cutilSafeCall(cudaMemset(newRgbMap, 0, m_width*m_height*sizeof(CAMERA_RGB_TYPE))); // sizeof(uint) == 4 // TODO MAYBE REDUNDANT. WE WRITE ON ALL IMAGE
			cutilSafeCall(cudaMemset(newDepthMap, 0, m_width*m_height*sizeof(uint))); // sizeof(uint) == 4

			// call CUDA kernel, writing results to PBO
			cameraHandler->cameraIteration(d_newDepthMap, newRgbMap, newDepthMap, newDepthData, d_currentTransform, newNormalMap, gridSize, blockSize);

			cutilCheckMsg("kernel failed");

			cutilSafeCall(cudaGraphicsUnmapResources(1, &pboArr[PBO_NEW_NORMAL].cuda_pbo_resource, 0));
			cutilSafeCall(cudaGraphicsUnmapResources(1, &pboArr[PBO_RGB].cuda_pbo_resource, 0));
			cutilSafeCall(cudaGraphicsUnmapResources(1, &pboArr[PBO_DEPTH].cuda_pbo_resource, 0));
		}

		if (!terminateTracking && startTracking) {
			cudaDeviceSynchronize();

			// call CUDA kernel, writing results to PBO
			icp_module->Iterate_ICP(newDepthData, d_renderedVertexMap, d_currentTransform);

			cutilCheckMsg("kernel failed");
		}

		if (startTracking) {
			General::Instance()->UpdateTransforms(d_currentTransform,
							 d_invCurrentTransform,
							 d_cameraPosition,
							 terminateTracking ? invViewMatrix : NULL);	// Pass invViewMatrix for traveling through the volume
			cudaDeviceSynchronize();
			//m_raycaster->CopyCameraPosition(d_cameraPosition); //volume render
			cudaDeviceSynchronize();

			// TODO - FOR TRAVELING THROUGH THE VOLUME WE MUST RESET THE VIEW VECTORS
			viewRotation = make_float3(0.f, 0.f, 0.f);
			viewTranslation = make_float3(0.f, 0.f, 0.f);
		}

		glutReportErrors();
		cutilCheckError(cutStopTimer(timer));
		computeFPS();
	}
}


void RunApp::mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
		buttonState  |= 1<<button;
	else if (state == GLUT_UP)
		buttonState = 0;

	ox = x; oy = y;
	glutPostRedisplay();
}

void RunApp::motion(int x, int y)
{
	if (terminateTracking) {
		float dx, dy;
		dx = (float)(x - ox);
		dy = (float)(y - oy);

		if (buttonState == 4) {
			// right = zoom
			viewTranslation.z += dy ;// / 100.0f;
		} 
		else if (buttonState == 2) {
			// middle = translate
			viewTranslation.x += dx; // / 100.0f;
			viewTranslation.y += dy; // / 100.0f;
		}
		else if (buttonState == 1) {
			// left = rotate
			viewRotation.x -= dy / 10.f; // / 5.0f;
			viewRotation.y += dx / 10.f; // / 5.0f;
		}

		ox = x; oy = y;
		glutPostRedisplay();
	}
}

void RunApp::reshape(int w, int h)
{
	if(initDone && !startTracking) {
		//m_width = w; m_height = h;		// YOU MUST NOT CHANGE WIDTH AND HEIGHT
		initPixelBuffer();

		// TODO - MAYBE DON'T PUT IT HERE. IT SHOULD BE A CONST
		// calculate new grid size
		gridSize = dim3(iDivUp(m_width, blockSize.x), iDivUp(m_height, blockSize.y));

		glViewport(0, 0, w, h);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		// left right bottom top zNear zFar
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); //top and bottom switched to make it like opencv
	}
}


void displayCallBack() {
	RunApp* instance = RunApp::thisClass;
	instance->display();
}

void motionCallBack(int x, int y) {
	RunApp* instance = RunApp::thisClass;
	instance->motion(x, y);
}

void mouseCallBack(int button, int state, int x, int y) {
	RunApp* instance = RunApp::thisClass;
	instance->mouse(button, state, x, y);
}

void reshapeCallBack(int w, int h) {
	RunApp* instance = RunApp::thisClass;
	instance->reshape(w, h);
}

void keyboardCallBack(unsigned char key, int x, int y) {
	RunApp* instance = RunApp::thisClass;
	instance->keyboard(key, x, y);
}

void idleCallBack() {
	RunApp* instance = RunApp::thisClass;
	instance->idle();
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int	main( int argc, char** argv)
{
	while(1) {
		RunApp runner(argc, argv);
	}

	return 0;
}

