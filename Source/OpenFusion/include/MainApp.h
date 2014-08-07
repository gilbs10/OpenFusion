#ifndef ___RUNAPP_H_
#define ___RUNAPP_H_

#define PBO_MAIN 0
#define PBO_DEPTH 1
#define PBO_NEW_NORMAL 2
#define PBO_RGB 3
#define PBO_FILE 4

#define NUM_OF_PBO 5

#define EMAIL_LENGTH 200

// CUDA Include
#include <vector_types.h>

#include "CameraHandler.h"
#include "ICP.h"
#include "VolumeIntegrator.h"
#include "Raycaster.h"

class RunApp {
	float3 viewRotation; 
	float3 viewTranslation;
	float invViewMatrix[12];
	
	// For Mouse Callback
	int ox, oy;
	int buttonState;

	// The pointer to the voxel array on the GPU
	VoxelType *d_voxel_array;
	VoxelWeightType *d_weight_array;

	ICP* icp_module;
	VolumeIntegrator* m_volumeIntegrator;
	CameraHandler* cameraHandler;
	Raycaster* m_raycaster;

	// This is copied to a symbol in the volumeRender_kernel module, and is used for the d_render function
	float3* d_cameraPosition; // This one is for the volume integration (TSDF parameter)

	PointInfo d_renderedVertexMap; // this holds the vertices and normals found during the raycasting.

	float* d_currentTransform; // Holds the current transform on cuda
	float* d_invCurrentTransform; // Holds the inverse current transform on cuda
	CAMERA_DEPTH_TYPE* d_newDepthMap;	// Holds the depth map coming from camera in each iteration (recycled memory)

	float* d_savedTransform; // Holds the transform used when tracking was enabled
	CameraData newDepthData; // Holds the new depth data and the calculated normals (global)

	PboData pboArr[NUM_OF_PBO];

	bool initDone;
	bool startTracking;
	bool writingModeOn;
	bool terminateTracking; // TODO - This is used to stop the tracking and allow traveling through the volume with the mouse
	bool showNormalsFlag;
	bool tPressed;
	int iteration;

	char emailAddress[EMAIL_LENGTH];
	int savesCounter;

	uint m_width;
	uint m_height;


	// Methods
	void PrintWelcomeMessage();
	void initCudaPointers();
	void cleanup();
	void render();
	void initPixelBuffer();
	void drawTextureToScreen(int buffer);
	void saveImageToFile();
	void initGL(int *argc, char **argv);
	int chooseCudaDevice(int argc, char **argv, bool bUseOpenGL);
	void saveVolumeToFile();

public:
	static RunApp* thisClass;

	void display();
	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
	void reshape(int w, int h);
	void keyboard(unsigned char key, int x, int y);
	void idle();

	RunApp(int argc, char** argv);

	~RunApp();
	void MainLoop();
};

#endif // ___RUNAPP_H_

