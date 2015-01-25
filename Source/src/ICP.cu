#include "utilities.h"
#include "CudaUtilities.cuh"
#include "ICP.h"

__device__ uint rgbaFloatToInt_ICP(float4 rgba)		// ALSO IN RENDERING!!!!!!!!
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

// TODO DEBUGGING
__device__ uint rgbaFloatToInt_debugging(float4 rgba) // TODO PAY ATTENTION: THIS IS ALSO ON volume render kernel.  we don't actually need it here, and it should be removed
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


/*__device__					// TODO PAY ATTENTION: THIS IS ALSO ON TSDF.CU!!! MAKE A GENERAL OPERATIONS FILE
__inline__ float4 multiply(const float* const mat4x4, const float4 vector){
	float res[4];
	for (int i=0; i<4; i++) {
		res[i] = dot(make_float4(mat4x4[i*4], mat4x4[i*4+1], mat4x4[i*4+2], mat4x4[i*4+3]), vector);
	}
	float d = 1.f / res[3];
	return make_float4(res[0] * d, res[1] * d, res[2] * d, 1.f);
}*/

/*__device__
__inline__ float3 multiply(const float* const mat3x3, const float3 vector) {
	float res[3];
	for (int i=0; i<3; i++){
		res[i] = dot(make_float3(mat3x3[i*3], mat3x3[i*3+1], mat3x3[i*3+2]), vector);
	}
	return make_float3(res[0], res[1], res[2]);
}*/

/*__device__ float3 getNormal (const uint x, 
							 const uint y,
							 const float3 currVertex, 
							 const float* const newDepthMap, 
							 const float* const transform, 
							 const CameraParams& cameraParams) { //TODO--------------------------------------------- REFERENCE!!!???!!!!       <- What????
	// currVertex is the new pixel's vertex in camera space (destVertexInCameraSpace)

	// from the transform matrix we extract the rotation matrix
	float rotationMatrix[] = { transform[0], transform[1], transform[2],
							   transform[4], transform[5], transform[6],
							   transform[8], transform[9], transform[10]};

	const uint c_Width = cameraParams.m_width;
	const uint c_Height = cameraParams.m_height;
	const Intrinsic& invIntrinsic = cameraParams.m_invIntrinsic;

	// Handling special cases of calculating normal in the last row/column of the array
	if ((x+1 == c_Width) || (y+1 == c_Height)) {
		if (x == 0) {

			float3 tempNormal = normalize( cross( ImageToCamera(invIntrinsic, make_float2(x, y-1), -newDepthMap[c_Width*(y-1)+(x)]) - currVertex,
												  ImageToCamera(invIntrinsic, make_float2(x+1, y), -newDepthMap[c_Width*(y)+(x+1)]) - currVertex));
			return normalize(multiply(rotationMatrix, tempNormal));
		}

		if (y == 0) {
			float3 tempNormal = normalize( cross( ImageToCamera(invIntrinsic, make_float2(x, y+1), -newDepthMap[c_Width*(y+1)+(x)]) - currVertex,
											      ImageToCamera(invIntrinsic, make_float2(x-1, y), -newDepthMap[c_Width*(y)+(x-1)]) - currVertex));
			return normalize(multiply(rotationMatrix, tempNormal));
		}

		float3 tempNormal = normalize( cross( ImageToCamera(invIntrinsic, make_float2(x-1, y), -newDepthMap[c_Width*(y)+(x-1)]) - currVertex,
											  ImageToCamera(invIntrinsic, make_float2(x, y-1), -newDepthMap[c_Width*(y-1)+(x)]) - currVertex));
		return normalize(multiply(rotationMatrix, tempNormal));

	}

	// Calculating the normal as explained in Depth Map Conversion chapter in Microsoft's paper
	float3 tempNormal = normalize( cross( ImageToCamera(invIntrinsic, make_float2(x+1, y), -newDepthMap[c_Width*(y)+(x+1)]) - currVertex ,
				                          ImageToCamera(invIntrinsic, make_float2(x, y+1), -newDepthMap[c_Width*(y+1)+(x)]) - currVertex));
	return normalize(multiply(rotationMatrix, tempNormal));
}*/


// PAY ATTENTION: THIS IS ALSO ON TSDF.CU!!! MAKE A GENERAL OPERATIONS FILE
__device__ __inline__
bool notInFrustum (const int x, const int y, const float z, uint c_Width, uint c_Height) {
												// TODO: MAYBE A BUG :) SHOULD WE CHECK FOR Z_FAR AND ELIMINATE IT?
												// PAY ATTENTION THAT IN CAMERA SPACE (AND Z-COORD IS COPIED FROM THERE) THE Z-COORD IS ALWAYS NEGATIVE, LIKE IN OPENGL
	return x < 0 || x >= c_Width || y < 0 || y >= c_Height || z >= -_EPSILON_;
}

// ALSO APPEARS IN TSDF.cu
__device__
float calculateAverage(float a, float b, float c, float d)
{
	int sum = 0;

	//Yuck. Counts the number of non-zero parameters participating in the average.  Ichs
	if(a > _EPSILON_)
		sum ++;
	if(b > _EPSILON_)
		sum++;
	if(c > _EPSILON_)
		sum ++;
	if(d > _EPSILON_)
		sum++;

	return (a+b+c+d)/sum;
}

__global__ 
void d_find_corresponding(const CameraData newDepthData,	// TODO - PAY ATTENTION THAT DEPTH MAP FROM CAMERA IS ALWAYS POSITIVE!!!!!!! (THATS WEIRD) MAYBE BUGGY :)
						  const PointInfo currentVertexMap,
						  float* A,
						  float* b,
						  float* AtA,
						  float* Atb,
						  const float* const invLastTransform,
						  const float* const newTransform,
						  const CameraParams cameraParams)
{
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= cameraParams.m_width) || (y >= cameraParams.m_height)) return;
	const uint offset = cameraParams.m_width*y + x;

	// Shared memory for storing AtA and Atb
	__shared__ float s_AtA[CORRESPONDENCE_BLOCK];
	__shared__ float s_Atb[CORRESPONDENCE_BLOCK];

	float A_row[NUM_VARS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	float b_row = 0.f;

	const float newDepth = newDepthData.depth[offset];
	const float3 currentVertex = currentVertexMap.vertex[offset];
	const float3 currentNormal = currentVertexMap.normal[offset];

	//If the depth at this pixel is undefined set row in A and b to zero
	//OpenCV's badCoord is zero.  We're checking a range of epsilon from zero.  Generally speaking, the value should be in milimeters and shouldn't be close to zero if it's okl. 
	if (!(fabs(newDepth) < cameraParams.m_min_depth ||
	  	  fabs(newDepth) > cameraParams.m_max_depth ||
		  currentVertex.x == BAD_VERTEX ||			// TODO: THERE'S A FUNCTION ISBADVERTEX
		  currentVertex.y == BAD_VERTEX ||
		  currentVertex.z == BAD_VERTEX)) {

		// v(i-1) = invT(i-1) * v(g)(i-1)
		float3 sourceVertexInCameraSpace = WorldToCamera(invLastTransform, currentVertex);

		// p = perspective project vertex v(i-1)
		float2 sourceVertexInImageSpace = CameraToImage(cameraParams.m_intrinsic, sourceVertexInCameraSpace);
	
		if(!(abs(sourceVertexInCameraSpace.z) < _EPSILON_)) {	// TODO: Just to make sure we don't divide by zero

			int newPixelColumn = roundFloat(sourceVertexInImageSpace.x);
			int newPixelRow = roundFloat(sourceVertexInImageSpace.y);
	
			int newOffset = newPixelRow*cameraParams.m_width + newPixelColumn; // The offset using the new pixel found

			// Pay attention that the z coordinate is taken from the vertex before projection (multiplied by (0,0,1) when multiplying by projection matrix)
			// In addition, we check that there's depth data in the new pixel (Di(p) > 0)
			if (! (notInFrustum(newPixelColumn, newPixelRow, sourceVertexInCameraSpace.z, cameraParams.m_width, cameraParams.m_height) ||
					newOffset < 0 ) ) { // ||
					//abs(newDepth) < CAMERA_DEPTH_MIN_MM || // TODO REMOVE?
					//abs(newDepth) > CAMERA_DEPTH_MAX_MM ) ) {


				float3 destVertexGlobal = newDepthData.vertex[newOffset];

				// TODO - CONSIDER CHECKING THAT THE DESTVERTEXGLOBAL IS INSIDE THE VOLUME

				// Calculate normals in new vertex map according to the depth map conversion section of Microsoft's paper.
				float3 newNormal = newDepthData.normal[newOffset];

				if ((length( destVertexGlobal - currentVertex ) < DISTANCE_THRESHOLD) &&
					(dot( newNormal, currentNormal ) > NORMAL_THRESHOLD) &&
					(dot( newNormal, currentNormal ) < (1.f + _EPSILON_) ) )
					// TODO: cosine of a small angle is close to 1 so the threshold needs to be greater than
				{
					
					// According to KinectShape
					A_row[0] =  -(currentNormal.z * destVertexGlobal.y -
									currentNormal.y * destVertexGlobal.z);

					A_row[1] =  -(currentNormal.x * destVertexGlobal.z -
									currentNormal.z * destVertexGlobal.x);

					A_row[2] =  -(currentNormal.y * destVertexGlobal.x -
									currentNormal.x * destVertexGlobal.y);

					A_row[3] = currentNormal.x;
					A_row[4] = currentNormal.y;
					A_row[5] = currentNormal.z;
		
					b_row = currentNormal.x * (currentVertex.x - destVertexGlobal.x) +
							currentNormal.y * (currentVertex.y - destVertexGlobal.y) +
							currentNormal.z * (currentVertex.z - destVertexGlobal.z);
				}
			}
		}
	}

	uint block_i = blockIdx.y * gridDim.x + blockIdx.x;
	uint thread_i = blockDim.x * threadIdx.y + threadIdx.x;

	// Sum and store AtA matrix values (only upper half of the symmetric matrix)
	for (int i = 0; i < NUM_VARS; i++) {
		for (int j = i; j < NUM_VARS; j++) {
			// Add up values one-by-one, once they are computed by all threads
			s_AtA[thread_i] = A_row[i] * A_row[j];
			__syncthreads();

			// Tree reduction
			for (unsigned int s = 1; s < CORRESPONDENCE_BLOCK; s *= 2) {
				if (thread_i % (2 * s) == 0) {
					s_AtA[thread_i] += s_AtA[thread_i + s];
				}
				__syncthreads();
			}

			// Store result in the global structure
			if (thread_i == 0) {
				AtA[block_i * (NUM_VARS*NUM_VARS) + i * NUM_VARS + j] = s_AtA[0];
			}
		}
	}

	// Do the same for Atb vector
	for (int i = 0; i < NUM_VARS; i++) {
		// Add up values one-by-one, once they are computed by all threads
		s_Atb[thread_i] = b_row * A_row[i];
		__syncthreads();

		// Tree reduction
		for (unsigned int s = 1; s < CORRESPONDENCE_BLOCK; s *= 2) {
			if (thread_i % (2 * s) == 0) {
				s_Atb[thread_i] += s_Atb[thread_i + s];
			}
			__syncthreads();
		}

		// Store result in the global structure
		if (thread_i == 0) {
			Atb[block_i * NUM_VARS + i] = s_Atb[0];
		}
	}
}


void ICP::FindCorresponding(const CameraData& newDepthData,
						    const PointInfo& currentVertexMap)
{
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(iDivUp(m_cameraParams.m_width, BLOCK_SIZE), iDivUp(m_cameraParams.m_height, BLOCK_SIZE));
		
	d_find_corresponding<<<numBlocks, threadsPerBlock>>>(newDepthData, currentVertexMap, d_A_Matrix, d_b_Matrix,
													AtA, Atb, invLastTransform, newTransform, m_cameraParams);
}


// oldTransform is the T(i-1), meaning currentTransform.
void ICP::Cpu_buildNewTransformMatrix (float* xOpt, float* newTransform, float* fullIncrementalTransform) {
	float alpha = xOpt[0];
	float beta = xOpt[1];
	float gamma = xOpt[2];
	float tx = xOpt[3];
	float ty = xOpt[4];
	float tz = xOpt[5];
	
	float incrementalTransform[16];
	incrementalTransform[0] = cos(gamma)*cos(beta);	//r11
	incrementalTransform[1] = -sin(gamma)*cos(alpha)+cos(gamma)*sin(beta)*sin(alpha);	//r12
	incrementalTransform[2] = sin(gamma)*sin(alpha)+cos(gamma)*sin(beta)*cos(alpha);	//r13
	incrementalTransform[3] = tx;
	incrementalTransform[4] = sin(gamma)*cos(beta);	//r21
	incrementalTransform[5] = cos(gamma)*cos(alpha)+sin(gamma)*sin(beta)*sin(alpha);	//r22
	incrementalTransform[6] = -cos(gamma)*sin(alpha)+sin(gamma)*sin(beta)*cos(alpha);	//r23
	incrementalTransform[7] = ty;
	incrementalTransform[8] = -sin(beta);	//r31
	incrementalTransform[9] = cos(beta)*sin(alpha);	//r32
	incrementalTransform[10] = cos(beta)*cos(alpha);	//r33
	incrementalTransform[11] = tz;
	incrementalTransform[12] = 0.f;
	incrementalTransform[13] = 0.f;
	incrementalTransform[14] = 0.f;
	incrementalTransform[15] = 1.f;

	float oldTransform[16];
	for (int i = 0; i < 16; ++i) {
		oldTransform[i] = newTransform[i];
		newTransform[i] = 0.f;
	}

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			for (int k = 0; k < 4; ++k) {
				//newTransform[i * 4 + j] += oldTransform[i * 4 + k] * incrementalTransform[k * 4 + j]; // TODO - THIS IS OUR BUG!!!!!!!! HELLLOOOO BUGGYYY!!!!!
				newTransform[i * 4 + j] += incrementalTransform[i * 4 + k] * oldTransform[k * 4 + j];
			}
		}
	}

	// Update the full incremental transform (accumulated incremental transform for this frame in all iterations)
	float full[16] = { 0.f };

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			for (int k = 0; k < 4; ++k) {
				full[i * 4 + j] += fullIncrementalTransform[i * 4 + k] * incrementalTransform[k * 4 + j];
			}
		}
	}

	for (int i = 0; i < 16; ++i) {
		fullIncrementalTransform[i] = full[i];
	}
}


void ICP::BuildNewTransformMatrix () {
	// TODO - MAKE THOSE PARAMETERS CLASS MEMBERS
	float xOpt_host[NUM_VARS];
	float newTransform_host[16];
	float incrementalTransform_host[16];

	cutilSafeCall(cudaMemcpy(xOpt_host, xOpt, sizeof(float)*NUM_VARS, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(newTransform_host, newTransform, sizeof(float)*16, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(incrementalTransform_host, incrementalTransform, sizeof(float)*16, cudaMemcpyDeviceToHost));

	// Launch on CPU
	Cpu_buildNewTransformMatrix(xOpt_host, newTransform_host, incrementalTransform_host);

	cutilSafeCall(cudaMemcpy(xOpt, xOpt_host, sizeof(float)*NUM_VARS, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(newTransform, newTransform_host, sizeof(float)*16, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(incrementalTransform, incrementalTransform_host, sizeof(float)*16, cudaMemcpyHostToDevice));
}
