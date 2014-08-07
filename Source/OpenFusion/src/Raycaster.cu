#include "Raycaster.h"
#include "utilities.h"
#include "CudaUtilities.cuh"


struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

// transform vector by matrix with translation
// M is 4x4
__device__
float4 mul(const float* M, const float4 &v)
{
    float4 r;
	float w = 1.f / dot(v, make_float4(M[12], M[13], M[14], M[15]));
    r.x = dot(v, make_float4(M[0], M[1], M[2], M[3])) * w;
    r.y = dot(v, make_float4(M[4], M[5], M[6], M[7])) * w;
    r.z = dot(v, make_float4(M[8], M[9], M[10], M[11])) * w;
	r.w = 1.f;

    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

__device__
int3 convertWorldToVoxels (float3 pos) {
	return make_int3( floor(( pos.x/(float)worldVolume+0.5f) * VOLUME_SIZE),
					  floor(( pos.y/(float)worldVolume+0.5f) * VOLUME_SIZE),
					  floor(( pos.z/(float)worldVolume+0.5f) * VOLUME_SIZE));
}

__inline__ __device__
float3 getRenderedNormal(const int3& curVoxel, VoxelType* d_voxel_array) {
	// Handling special cases of calculating normal in the last row/column of the array
	int x = curVoxel.x;
	int y = curVoxel.y;
	int z = curVoxel.z;

	//Aaron's idea for getting more accurate normals:  He thinks the Raycasting should be done in two phases.  In the first we find the surfacePos for each pixel.  In the next phase we send the resulting array
	//as input to the function.  We go over the array, and for each surfacePos we look at the neighboring pixels and calculate the normal like we do for the image in ICP, only we calculate the normal for each of the neighbors
	//and then interpolate them to find the actual normal.  This is similar to how you find normals in a triangle mesh.
	
	
	if (x <= 0 || (x+1 >= VOLUME_SIZE) || y <= 0 || (y+1 >= VOLUME_SIZE) || z <= 0 || (z+1 >= VOLUME_SIZE)) {
		// TODO WE HAVE DECIDED TO IGNORE THE VOXELS THAT ARE ON FACES, BECAUSE CALCULATING THE 6 NEIGHBORS FOR THEM IS LONG AND ARDOUS (BAHHHH) MAYBE BUGGY!
		// WE CURRENTLY IGNORE THEM BUT ITS REALLY EASY TO SUPPORT IT TOO! :) THERE ARE ONLY TWO OPTIONS FOR EACH COORDINATE (ONLY 6 CASES)
		// THIS CAN BE DONE ALONE
			return make_float3(0.f, 0.f, 0.f);
	}

	// The next section is from 10/08/2012, according to ISMAR

	#define VOLUME_INDEX(x,y,z) ((x)*VOLUME_SIZE*VOLUME_SIZE + (y)*VOLUME_SIZE + (z))

	float3 result = make_float3(d_voxel_array[VOLUME_INDEX(x+1, y, z)].tsdf - d_voxel_array[VOLUME_INDEX(x-1, y, z)].tsdf ,
								d_voxel_array[VOLUME_INDEX(x, y+1, z)].tsdf - d_voxel_array[VOLUME_INDEX(x, y-1, z)].tsdf ,
								d_voxel_array[VOLUME_INDEX(x, y, z+1)].tsdf - d_voxel_array[VOLUME_INDEX(x, y, z-1)].tsdf );
	#undef VOLUME_INDEX

	// If the vector was the zero vector, it can't be normalized (the function takes care of it)
	return normalize(result);
}


__global__ void
d_raycast(uint *d_output, float* d_transform, VoxelType *d_voxel_array, const PointInfo d_renderedVertexMap, CameraParams cameraParams, const float3* d_cameraPosition, bool showNormal)
{
	uint imageW = cameraParams.m_width;
	uint imageH = cameraParams.m_height;

	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

	const long offset = y*imageW + x;

	const int maxSteps = 850; //TODO: BE SMARTER, CHANGE THIS. Was 500. BE MUCH SMARTER!!!!!!!!!!!!!!!!!!! THEN IT WAS 1200
    const float tstep = 1.5f; // WAS 3.f at June 2013 // WAS 1.5f; // Was 0.01f (ONCE WAS 0.05 and steps were 800, before changing the box into world coordinates)

	const float zPlane = -1.f;
	//the size of the volume on screen.
    const float3 boxMin = make_float3(VOLUMEMINX, VOLUMEMINY, VOLUMEMINZ);
    const float3 boxMax = make_float3(VOLUMEMAXX, VOLUMEMAXY, VOLUMEMAXZ);

	// Transform image coordinates to camera space
	float3 cameraCoords = ImageToCamera(cameraParams.m_invIntrinsic, make_float2(x, y), zPlane);

   // calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = *d_cameraPosition;
	eyeRay.d = make_float3(cameraCoords.x, cameraCoords.y, zPlane); // Position in camera space of pixel (x,y) with depth = zPlane
    eyeRay.d = CameraToWorld(d_transform, eyeRay.d); // Transform this vertex into world space.
	eyeRay.d = normalize(eyeRay.d - eyeRay.o);
	
    // Find intersection with box, return if there is none
	float tnear, tfar;
	const int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit)
	{
		d_renderedVertexMap.vertex[offset] = make_float3(BAD_VERTEX);	// TODO MIRRORING THE IMAGE IN BOTH AXES: X AND Y
		d_renderedVertexMap.normal[offset] = make_float3(0.f);

		return;
	}
	if (tnear < 0.0f) tnear = 0.0f; // clamp to near plane

    float t = tnear;
    const float3 start = eyeRay.o + eyeRay.d*tnear;
	float3 pos = start;
	const int3 badVoxel = make_int3(-1, -1, -1);
	int3 prevVoxel = badVoxel;
	int3 curVoxel = badVoxel;
	float curVoxelValue = BAD_VERTEX;
    float3 step = eyeRay.d*tstep;
	float3 surfacePos = make_float3(BAD_VERTEX);
	bool surfaceFlag = false;

	int n;
    for(n=1; n<maxSteps; n++) {
		
		// Starting from one step makes makes sure that we're inside the box (and not a step behind)
		pos = start + n*step;

		// curVoxel holds the indices in the volume of the voxel we are currently in.  we move to [0,1] then multiply by VOLUME_SIZE to get [0,VOLUME_SIZE]

		// TODO - SAME CODE APPEARS BELOW
		curVoxel = convertWorldToVoxels(pos);

		// Out of array bounds - TODO: WE CAN JUST RETURN AND COLOR IT BLACK
		if(curVoxel.x < 0 || curVoxel.x >= VOLUME_SIZE || curVoxel.y < 0 || curVoxel.y >= VOLUME_SIZE || curVoxel.z < 0 || curVoxel.z >= VOLUME_SIZE) {
			break;
		}

		// This is the value of the TSDF in the current voxel
		curVoxelValue = d_voxel_array[curVoxel.x*VOLUME_SIZE*VOLUME_SIZE + curVoxel.y*VOLUME_SIZE + curVoxel.z].tsdf;
	
		if (curVoxelValue <= 0.f) {	// the current voxel is the zero-crossing
			// pos contains a vertex in the volume [-worldVolume/2, -..., -...]:[worldVolume/2, ..., ...]
			
			if (prevVoxel.x == badVoxel.x && prevVoxel.y == badVoxel.y && prevVoxel.z == badVoxel.z) { 
				// This means that the surface is touching the front wall of the volume.
				//  In this case we don't have an actual surface, so we use the depth of the vertex on the volume wall.
				surfacePos = pos; // That's an approximation
			}
			else {
				// prevVoxel is a voxel inside the box
				// CANCELLED! - NO TRILINEAR INTERPOLATION
				surfacePos = pos;//getSurfaceVertex(pos - step, pos, prevVoxel, curVoxel, d_voxel_array);	// Calls the trilinear interpolation

				// TODO - SAME CODE APPEARS ABOVE
				curVoxel = convertWorldToVoxels(surfacePos);
			}
			surfaceFlag = true;
			break;
		}

		prevVoxel.x = curVoxel.x;
		prevVoxel.y = curVoxel.y;
		prevVoxel.z = curVoxel.z;

        t += tstep;
        if (t > tfar - 2.0) break; // TODO CHANGE THIS MAGIC (HORROR) NUMBER

		// Weight handling


    }

	d_renderedVertexMap.vertex[offset] = surfacePos;

	float3 normal = make_float3(0.f);
	if(surfaceFlag)
	{
		normal = getRenderedNormal(curVoxel, d_voxel_array);
	}

	// TODO IF NORMAL IS THE ZERO VECTOR, MAYBE ITS BETTER TO TAKE ITS NEIGHBOR NORMAL, WHICH IS NOT ZERO (ZERO CAN CAUSE PROBLEMS)
	// CALCULATING THE NORMALS IS DONE BY CROSS VECTORS (AND THE DIRECTION IS THEREFORE UNKNOWN)
	d_renderedVertexMap.normal[offset] = normal;

	if(showNormal)
	{
		float3 temp = (normal+1.0)*0.5;
		d_output[offset] = rgbaFloatToInt(make_float4(temp)); // TODO THIS IS TEMPORARY. WE SWITCHED THE INDICES AGAIN (y*imageW + (imageW-1 - x))
	}
	else
	{
		float tempDot = dot(-eyeRay.d, normal);
		if(fabs(tempDot) < _EPSILON_)
			d_output[offset] = rgbaFloatToInt(make_float4(0.f));
		else {
			d_output[offset] = rgbaFloatToInt(make_float4(make_float3(0.6 * tempDot + 0.2)));

		}
	}

}

//////////////////////////// CPP functions //////////////////////////////////

void Raycaster::Raycast(uint *d_output, float* transform, VoxelType* d_voxel_array, const PointInfo& d_renderedVertexMap, const float3* d_cameraPosition, bool showNormals)
{
	d_raycast<<<m_numBlocks, blockSize>>>( d_output, transform, d_voxel_array, d_renderedVertexMap, m_cameraParams, d_cameraPosition, showNormals);
}

Raycaster::Raycaster(CameraParams* cameraParams) : m_cameraParams(*cameraParams)
{
	m_numBlocks = dim3(iDivUp(m_cameraParams.m_width, blockSize.x), iDivUp(m_cameraParams.m_height, blockSize.y));
}

/////////////////////////////Maybe use later: //////////////////////////////////////

/*
// The function returns true if the voxel is out of bounds 
__inline__ __device__ bool voxelOutOfBounds (int3 indices) {
	return  indices.x < 0 || indices.x >= VOLUME_SIZE ||
			indices.y < 0 || indices.y >= VOLUME_SIZE ||
			indices.z < 0 || indices.z >= VOLUME_SIZE;
}


// The function returns the value of the indicated voxel. If voxel is out of bounds then we return 0. If the voxel exists we increase the count
__inline__ __device__ float getTSDFValue (int3 indices, uint* count, VoxelType* d_voxel_array) {
	if (voxelOutOfBounds(indices)) {
		return 0.0f;
	}
	*count++;
	return d_voxel_array[indices.x*VOLUME_SIZE*VOLUME_SIZE + indices.y*VOLUME_SIZE + indices.z].tsdf;
}


// The function gets indices of a voxel and returns the averaged tsdf value in its FRONT TOP LEFT point 
__device__ float weightedAverageInLatticePoint (uint3 indices, VoxelType* d_voxel_array) {
	float totalValue = 0.0f;
	uint countNeighbors = 0;
	uint i = indices.x, j = indices.y, k = indices.z;
	
	totalValue += getTSDFValue (make_int3(i    , j    , k    ), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i - 1, j    , k    ), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i    , j + 1, k    ), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i - 1, j + 1, k    ), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i    , j    , k + 1), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i - 1, j    , k + 1), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i    , j + 1, k + 1), &countNeighbors, d_voxel_array);
	totalValue += getTSDFValue (make_int3(i - 1, j + 1, k + 1), &countNeighbors, d_voxel_array);

	if (countNeighbors == 0) {
		return 0.0f;
	}
	return totalValue/countNeighbors;
}

//receives a vertex and uses trilinear interpolation. Returns the TSDF value in this vertex
__device__ float trilinearInterpolation(float3 vertex, int3 indices, VoxelType* d_voxel_array) {
	//http://en.wikipedia.org/wiki/Trilinear_interpolation

	uint i = indices.x, j = indices.y, k = indices.z;

	//(x_0, y_0, z_0) represents the minimum vertex of the given voxel. (lower back left)
	float x_0 = VOLUMEMINX + (VOLUMEWIDTH/VOLUME_SIZE)*indices.x;
	float x_d = (vertex.x - x_0) / (VOLUMEWIDTH/VOLUME_SIZE);

	// c(xyz) holds the value in the xyz vertex of the lattice
	// meanwhile we interpolate the values along x, which is held in c(yz)
	float c000 = weightedAverageInLatticePoint (make_uint3(i  ,j-1, k-1), d_voxel_array);
	float c100 = weightedAverageInLatticePoint (make_uint3(i+1,j-1, k-1), d_voxel_array);
	float c00 = c000*(1-x_d) + c100*(x_d);

	float c010 = weightedAverageInLatticePoint (make_uint3(i  ,j  , k-1), d_voxel_array);
	float c110 = weightedAverageInLatticePoint (make_uint3(i+1,j  , k-1), d_voxel_array);
	float c10 = c010*(1-x_d) + c110*(x_d);
	
	float c001 = weightedAverageInLatticePoint (make_uint3(i  ,j-1, k  ), d_voxel_array);
	float c101 = weightedAverageInLatticePoint (make_uint3(i+1,j-1, k  ), d_voxel_array);
	float c01 = c001*(1-x_d) + c101*(x_d);
	
	float c011 = weightedAverageInLatticePoint (make_uint3(i  ,j  , k  ), d_voxel_array);
	float c111 = weightedAverageInLatticePoint (make_uint3(i+1,j  , k  ), d_voxel_array);
	float c11 = c011*(1-x_d) + c111*(x_d);
	
	

	float y_0 = VOLUMEMINY + (VOLUMEWIDTH/VOLUME_SIZE)*indices.y;
	float y_d = (vertex.y - y_0) / (VOLUMEWIDTH/VOLUME_SIZE);

	// Then we interpolate these values along y
	float c0 = c00*(1-y_d) + c10*(y_d);
	float c1 = c01*(1-y_d) + c11*(y_d);

	float z_0 = VOLUMEMINZ + (VOLUMEWIDTH/VOLUME_SIZE)*indices.z;
	float z_d = (vertex.z - z_0) / (VOLUMEWIDTH/VOLUME_SIZE);

	// Finally we interpolate these values along z
	float c = c0*(1-z_d) + c1*(z_d);

	return c; // Finally
}

// The function receives vertexes on either side of the zero crossing and the voxels they reside in 
// and returns the vertex at the zero crossing
__device__ float3 getSurfaceVertex(float3 prevPos, float3 nextPos, int3 prevVoxel, int3 nextVoxel, VoxelType* d_voxel_array) {
	// Assuming prevPos' value <0, nextPos' value >0
	
	float prevValue = trilinearInterpolation(prevPos, prevVoxel, d_voxel_array);
	float nextValue = trilinearInterpolation(nextPos, nextVoxel, d_voxel_array);
	
	// Validation
	if (prevValue < 0.f || nextValue > 0.f) {
		return make_float3(VOLUMEMINX-1, VOLUMEMINY-1, VOLUMEMINZ-1);
	}

	//if the values are close return the midway point, otherwise linearly interpolate the actual position
	if(prevValue - nextValue < _EPSILON_) {
		return 0.5*(prevPos + nextPos);
	}
	 return prevPos + (-nextValue)/(prevValue - nextValue)*(nextPos-prevPos);
}
*/
