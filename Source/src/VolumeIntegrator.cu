#include "utilities.h"
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>

#include "VolumeIntegrator.h"
#include "CudaUtilities.cuh"


//__constant__ float3 c_CameraPosition;

// PAY ATTENTION: THIS IS ALSO (NOT EXACTLY THE SAME ON volumeRender_ICP.CU!!! MAKE A GENERAL OPERATIONS FILE
__device__
__inline__
bool notInFrustum (int2 vector, float z, uint width, uint height, float camera_min_depth, float camera_max_depth) {
	return vector.x < 2 || vector.x >= width-2 || vector.y < 2 || vector.y >= height-2 || fabs(z) > camera_max_depth || fabs(z) < camera_min_depth;
}



__device__					// PAY ATTENTION: THIS IS ALSO ON volumeRender_ICP.CU!!! MAKE A GENERAL OPERATIONS FILE
__inline__ float4 multiply(const float* const mat4x4, float4 vector) {
	float res[4];
	for (int i=0; i<4; i++){
		res[i] = dot(make_float4(mat4x4[i*4], mat4x4[i*4+1], mat4x4[i*4+2], mat4x4[i*4+3]), vector);
	}
	float d = 1.f / res[3];
	return make_float4(res[0] * d, res[1] * d, res[2] * d, 1.f);
}

__device__					// PAY ATTENTION: THIS IS ALSO ON volumeRender_ICP.CU!!! MAKE A GENERAL OPERATIONS FILE
__inline__ float3 multiply(const float* const mat3x3, float3 vector){
	float res[3];
	for (int i=0; i<3; i++) {
		res[i] = dot(make_float3(mat3x3[i*3], mat3x3[i*3+1], mat3x3[i*3+2]), vector);
	}
	return make_float3(res[0],res[1],res[2]);
}


__device__
float3 convertVoxelsToWorld (uint3 indices) {
	float ratio = (float)worldVolume / VOLUME_SIZE;
	float halfVolume = (float)worldVolume * 0.5f;
	return make_float3(  indices.x * ratio - halfVolume + ratio/2,
					     indices.y * ratio - halfVolume + ratio/2,
					     indices.z * ratio - halfVolume + ratio/2 );
		// For example, voxelVolume is 0...512 (indices) and the world is -512...512. Then we translate the result into the center of the voxel by adding 1 to each coordinate
}

__device__
__inline__
float floatMin (float a, float b) {
	return a<b ? a : b;
}

__device__
__inline__
float floatMax (float a, float b) {
	return a>b ? a : b;
}

__device__
float calculateAverageTSDF(float a, float b, float c, float d)
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

__device__ __forceinline__ bool isBadVertex (float3 vertex) {	// TODO - APPEARS ALSO IN CAMERAINPUT.CU
	//return (vertex.x <= BAD_VERTEX || vertex.y <= BAD_VERTEX || vertex.z <= BAD_VERTEX);
	return (vertex.x == BAD_VERTEX) || (vertex.y == BAD_VERTEX) || (vertex.z == BAD_VERTEX);
}

__global__
void d_integrateVolume (VoxelType* d_voxel_array, 
						VoxelWeightType* d_weight_array, 
						const float* invTransform, 
						const CameraData d_depthData,
						const CameraParams cameraParams,
						const float3* d_cameraPosition) {

	const int z = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( y >= VOLUME_SIZE || z >= VOLUME_SIZE) {
		return;
	}

	long offset;
	float3 worldVector;
	float3 cameraVector;
	int2 projectedVector;
	float3 depthMapConvertedToWorld;
	float SDF;
	float nextTSDF;
	float previousTSDF;
	VoxelWeightType previousWeight;
	const VoxelWeightType weight = 1;

	//TODO - temporary variable.  make cameraToImage return uint2 instead of float2.  Needs to be dealt with in all modules.
	float2 projectedVectorFloat;

	const long partial_offset = y*VOLUME_SIZE + z;
	offset = partial_offset;
	const long step = VOLUME_SIZE*VOLUME_SIZE;

	for (int x = 0; x < VOLUME_SIZE; ++x) {
		
		//offset = x*VOLUME_SIZE*VOLUME_SIZE + y*VOLUME_SIZE + z;
		offset += step;	// Advance offset by a single x


		worldVector = convertVoxelsToWorld(make_uint3(x,y,z));	// voxel [x,y,z] is being converted to the vertex it represents in world coordinates
		cameraVector = WorldToCamera(invTransform, worldVector); //          make_float3(multiply(invTransform, make_float4(worldVector, 1.0))); // transforming the vector to camera space
		projectedVectorFloat = CameraToImage(cameraParams.m_intrinsic, cameraVector); //    make_int2(perspectiveProject(cameraVector));
		projectedVector.x = roundFloat(projectedVectorFloat.x);
		projectedVector.y = roundFloat(projectedVectorFloat.y);

		if (notInFrustum(projectedVector, cameraVector.z, cameraParams.m_width, cameraParams.m_height, cameraParams.m_min_depth, cameraParams.m_max_depth)) {
			continue;
		}

		
		depthMapConvertedToWorld = d_depthData.vertex[projectedVector.y*cameraParams.m_width + projectedVector.x];

		if (isBadVertex(depthMapConvertedToWorld)) {
			continue;
		}
		
		// Taking care of the TSDF in this voxel. Updates ONLY if the new tsdf is not beyond a surface.
		previousTSDF = d_voxel_array[offset].tsdf;
		previousWeight = d_weight_array[offset];

		SDF = length(depthMapConvertedToWorld - *d_cameraPosition) - length(worldVector - *d_cameraPosition);

		if (SDF > 0) {
			// TODO - in KinectShape their range is not limited to [-1, 1]. It's between [-Truncation=-50, Truncation=50].
			nextTSDF = floatMin(SDF, TRUNCATION);
		}
		else {
			if (SDF >= -TRUNCATION) {
				nextTSDF = SDF;
			}
			else
			{
				continue;	// No TSDF Update
			}
		}

		// TODO - KinectShape suggests different weighted average: The new tsdf always gets weight == 1, as
		// opposed to what is written in the article. WE CHANGED THIS ACCORDING TO KINECTSHAPE.

		d_voxel_array[offset].tsdf = (previousTSDF*previousWeight + nextTSDF*weight) / (float)(previousWeight + weight);
		d_weight_array[offset] = min((VoxelWeightType)MAX_WEIGHT, previousWeight + 1);

		// 20121117 - WE STOPPED HERE (ACTUALLY NOT).
		// 1. WE GET TRAILING RAY ALONG THE FRUSTUM. IT LOOKS LIKE FLOOR BUT IT'S NOT.
		// 2. THINGS DISAPPEAR AFTER THEY DON'T SHOW IN THE FRUSTUM FOR A WHILE.
		// 3. TRY TO VIDEO ANOTHER DOLL. MAKE A ROUND OF 360 DEGREES TO SEE IF YOU GET THE UNIT MATRIX.
		//    THE NORMALS HAVE PROBLEMS AT THE "SILHOUETTE" OF THE MODEL.    <---------- OUR BIGGEST PROBLEM RIGHT NOW
		// 4. OPTIMIZATION - DEPTH CONVERSION.
		// 5. THE VOLUME IS STILL UPSIDE DOWN. WHEN WE SWITCHED IT THE TRACKING WAS LESS STABLE.
		// 6. SHOW THE ORIGINAL INPUT BESIDE THE VOLUME.
		// 7. REFACTOR
		// 8. MAINTAIN ERROR (AND ERROR SUM). IF THE ERROR IS GREATER THAN SOME THRESHOLD - ALSO DON'T UPDATE THE TSDF.
		//    IN ADDITION, YOU CAN CHANGE THE TRANSFORMATION BUT KEEP THE LAST DEPTH IMAGE AND TRY TO MATCH IT WITH THE CURRENT NEW TRANSFORM.
		//    TRY TO RECOVER FROM THOSE SINGULARITIES!
		// 9. DRAW THE CORRESPONDENCES SO YOU WILL KNOW WHAT HAPPENS IN FRAME ~#650.
		// 10. Remove the double zero crossing and change the weight of the negative part so that it changes more readily.
		// 11. Check KinectShape - they did some things only when weight had reached 10, maybe that's a clue.

		// 20121122 - Vered: I've changed the way the weight is being updated for the second zero crossing (not updated). That
		// seems to solve the holes, but still the image is shaking when we scan the model from behind.
	}
}



//////////////////////////////////////Host Functions: /////////////////////////////////////////////

VolumeIntegrator::VolumeIntegrator(CameraParams* cameraParams) : 
	m_cameraParams(*cameraParams), 
	m_numBlocks(iDivUp(VOLUME_SIZE, blockSize.x),iDivUp(VOLUME_SIZE, blockSize.y)){
	// empty
}

void VolumeIntegrator::Integrate (VoxelType* d_voxel_array, VoxelWeightType* d_weight_array, const float* invTransform, const CameraData& d_depthData, const float3* d_cameraPosition) {
	d_integrateVolume<<<m_numBlocks, blockSize>>>(d_voxel_array, d_weight_array, invTransform, d_depthData, m_cameraParams, d_cameraPosition);
}

