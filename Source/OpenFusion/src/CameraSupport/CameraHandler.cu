// TODO - WE STOPPED HERE (31/8/2012)
// WE NEED TO COPY THE DEPTH MAP (FROM CAMERA) TO CUDA. EACH depth data is a 2-byte uint value.  We want to copy an array of these into cuda, then use them to create an array
//	of floating point numbers which hold the same value (the depth data from the camera).  While we're at it we should change the y direction so that it fits our program
//	(opencv counts y in the opposite direction).
//
//	Apply bilateral filter (from cuda sdk examples).
#include "utilities.h"
#include "CameraHandler.h"

#include "CudaUtilities.cuh"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>

CameraParams CameraHandler::m_params;

// transform vector by matrix (no translation)
__device__
float3 mulCamera(const float* M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M[0], M[1], M[2]));
    r.y = dot(v, make_float3(M[3], M[4], M[5]));
    r.z = dot(v, make_float3(M[6], M[7], M[8]));
    return r;
}

// transform vector by matrix with translation
// M is 4x4
__device__
float4 mulCamera(const float* M, const float4 &v)
{
    float4 r;
	float w = 1.f / dot(v, make_float4(M[12], M[13], M[14], M[15]));
    r.x = dot(v, make_float4(M[0], M[1], M[2], M[3])) * w;
    r.y = dot(v, make_float4(M[4], M[5], M[6], M[7])) * w;
    r.z = dot(v, make_float4(M[8], M[9], M[10], M[11])) * w;
	r.w = 1.f;

    return r;
}

__device__ __inline__ bool isBadVertex (float3 vertex) {
	//return (vertex.x == BAD_VERTEX || vertex.y == BAD_VERTEX || vertex.z == BAD_VERTEX);
	return (vertex.x == BAD_VERTEX) || (vertex.y == BAD_VERTEX) || (vertex.z == BAD_VERTEX);
}

__device__ uint rgbaFloatToIntCamera(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

// d_newDepthMap == NULL iff the call is from ICP, updating the point only
__global__ void d_convertDepthToWorldVertices(const CameraData d_cameraData, CAMERA_DEPTH_TYPE* d_newDepth, float* transform, uint* newDepthMap, CameraParams cameraParams) {
	
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= cameraParams.m_width) || (y >= cameraParams.m_height)) return;
	const long offset = cameraParams.m_width*y+(x);

	// If we got a new depth map (d_newDepth != NULL) we update the corresponding value in d_cameraData and return the new value.
	// Else, we return the existing value.
	const CAMERA_DEPTH_TYPE newDepth = (d_newDepth != NULL) ? (d_cameraData.depth[offset] = d_newDepth[offset]) : d_cameraData.depth[offset];
	if (newDepthMap) {
		float newDepthNormalized = (float)newDepth/cameraParams.m_max_depth;
		newDepthMap[offset] = rgbaFloatToIntCamera(make_float4(newDepthNormalized, newDepthNormalized, newDepthNormalized, 1.f));
	}

	if ((abs(newDepth) < cameraParams.m_min_depth ||		// TODO: newDepth is possibly non-negative. Maybe this fabs is redundant
		 abs(newDepth) > cameraParams.m_max_depth)) {
		// No depth at point
		d_cameraData.vertex[offset] = make_float3(BAD_VERTEX);
	}
	else {
		// Calculate vertex coordinates
		d_cameraData.vertex[offset] = ImageToWorld(cameraParams.m_invIntrinsic, make_float2(x,y), -newDepth, transform);
	}
}

__global__ void d_convertDepthToWorldNormals(const CameraData d_cameraData, uint* newNormalMap, CameraParams cameraParams)
{
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= cameraParams.m_width) || (y >= cameraParams.m_height)) return;
	const long offset = cameraParams.m_width*y+(x);

	// Current vertex in world space
	float3 currVertex = d_cameraData.vertex[offset];
	float3 normal = make_float3(0.f);

	if (!isBadVertex(currVertex)) {
		float3 firstVertex = make_float3(0.f);
		float3 secondVertex = make_float3(0.f);

		// Handling special cases of calculating normal in the last row/column of the array
		if ((x+1 >= cameraParams.m_width) || (y+1 >= cameraParams.m_height)) {
			
			firstVertex = make_float3(BAD_VERTEX);
			secondVertex = make_float3(BAD_VERTEX);
		}
		else {
			// Calculating the normal as explained in Depth Map Conversion chapter in Microsoft's paper

			firstVertex = d_cameraData.vertex[offset + 1];	// Means (x+1, y). Might result in bad vertex
			secondVertex = d_cameraData.vertex[offset + cameraParams.m_width]; // Means (x, y+1)

			if (!isBadVertex(firstVertex) && !isBadVertex(secondVertex)) { // And of course (isBadVertex(currVertex) ==  False)
				// None of the above is bad vertex
				normal = normalize(cross(firstVertex - currVertex , secondVertex - currVertex));
			}
		}
	}
	// else: We don't have a depth data at that point. Therefore the normal is still 0,0,0
		
	d_cameraData.normal[offset] = normal;
	
	if(newNormalMap) {
		//float3 value = make_float3(cameraData[y*IMAGE_W+x].point.vertex.z);
		newNormalMap[y*cameraParams.m_width+x] = rgbaFloatToIntCamera(make_float4((normal+1.0)*0.5, 1.f));
	}
}

void CameraHandler::updateCameraData(dim3 gridSize, dim3 blockSize, const CameraData& d_cameraData, CAMERA_DEPTH_TYPE* d_newDepth, float* d_currentTransform, uint* newNormalMap, uint* newDepthMap)
{
	d_convertDepthToWorldVertices<<<gridSize, blockSize>>>(d_cameraData, d_newDepth, d_currentTransform, newDepthMap, m_params);
	cutilSafeCall(cudaDeviceSynchronize());
	
	d_convertDepthToWorldNormals<<<gridSize, blockSize>>>(d_cameraData, newNormalMap, m_params);
	cutilSafeCall(cudaDeviceSynchronize());

	// TODO: THINK HOW TO DO THIS BETTER. DIVIDE THE THREADS INTO FEWER BLOCKS TO SAVE RESOURCES.
}