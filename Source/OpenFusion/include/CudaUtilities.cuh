#ifndef ___CUDA_UTILITIES_H_
#define ___CUDA_UTILITIES_H_

#include "utilities.h"
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>


__device__ __inline__
float2 CameraToImage (const Intrinsic intrinsic, const float3 vec) {
	float2 retVal;

	if(fabs(vec.z) < _EPSILON_){ //TODO - remove if this is checked elsewhere
		return make_float2(BAD_VERTEX, BAD_VERTEX);
	}

	float invZ = 1.f / vec.z;
	retVal.x = (intrinsic.m_fx*vec.x + intrinsic.m_cx*vec.z)*invZ;
	retVal.y = (intrinsic.m_fy*vec.y + intrinsic.m_cy*vec.z)*invZ;

	return retVal;
}

__device__ __inline__
float3 ImageToCamera (const Intrinsic invIntrinsic, const float2 pixel, const float depth) {
	float3 retVal;

	retVal.x = depth*(invIntrinsic.m_fx*pixel.x + invIntrinsic.m_cx);
	retVal.y = depth*(invIntrinsic.m_fy*pixel.y + invIntrinsic.m_cy);
	retVal.z = depth;

	return retVal;
}

__device__ __inline__
float3 CameraToWorld (const float* transform, const float3 vec) {
	float3 retVal;

	retVal.x = transform[0]*vec.x + transform[1]*vec.y + transform[2]*vec.z + transform[3];
	retVal.y = transform[4]*vec.x + transform[5]*vec.y + transform[6]*vec.z + transform[7];
	retVal.z = transform[8]*vec.x + transform[9]*vec.y + transform[10]*vec.z + transform[11];

	return retVal;
}

__device__ __inline__
float3 WorldToCamera (const float* invTransform, const float3 vec) {
	return CameraToWorld(invTransform, vec);
}

__device__ __inline__
float3 ImageToWorld (const Intrinsic invIntrinsic, const float2 pixel, const float depth, const float* transform){
	return CameraToWorld(transform, ImageToCamera(invIntrinsic, pixel, depth));
}

__device__ __inline__
float2 WorldToImage (const float* invTransform, const float3 vec, const Intrinsic intrinsic){
	return CameraToImage(intrinsic, WorldToCamera(invTransform, vec));
}

//rounds the floating point to the nearest integer.
__device__ __inline__ int roundFloat(float a){
	return fracf(a) < 0.5 ? floorf(a) : ceilf(a);
}

#endif // ___CUDA_UTILITIES_H_