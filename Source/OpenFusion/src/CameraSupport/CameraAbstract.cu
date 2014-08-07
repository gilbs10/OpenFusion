#include "CameraAbstract.h"
#include "utilities.h"

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_math.h>


__device__ __inline__
float d_ComputeAverageNeighborhood(const CAMERA_DEPTH_TYPE* d_newDepthMap, const uint x, const uint y, const uint width, const uint height) {
	float sum = 0;
	for (int i = -1; i <= 1; ++i) {
		for (int j = -1; j <= 1; ++j) {
			if (i == 0 && j == 0) {
				continue;
			}
			sum += d_newDepthMap[width*((y + j) % height) + ((x + i) % width)];
		}
	}
	return sum/8.f;
}

__global__ void d_AverageFilter(const CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_DEPTH_TYPE* d_resultDepthMap, const uint width, const uint height, CAMERA_DEPTH_TYPE depthNullValue) {
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= width) || (y >= height)) return;
	const uint offset = width*y + x;
	const float threshold = 15.f;

	CAMERA_DEPTH_TYPE newValue = d_newDepthMap[offset];
	const float average = d_ComputeAverageNeighborhood(d_newDepthMap, x, y, width, height);
	newValue = (abs(average - newValue) > threshold) ? depthNullValue : average;

	d_resultDepthMap[offset] = newValue;
}

__device__ __inline__
float2 d_ComputeGradientByPrewitt(const CAMERA_DEPTH_TYPE* d_newDepthMap, const uint x, const uint y, const uint width, const uint height) {
	float2 gradient = make_float2(0.f);
	for (int i = -1; i <= 1; i += 2) {
		for (int j = -1; j <= 1; ++j) {
			gradient.x += i * d_newDepthMap[width*((y + j) % height) + ((x + i) % width)];
		}
	}

	for (int j = -1; j <= 1; j += 2) {
		for (int i = -1; i <= 1; ++i) {
			gradient.y += j * d_newDepthMap[width*((y + j) % height) + ((x + i) % width)];
		}
	}

	// TODO - DEBUG THIS CODE!!! PROBABLY WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


	return gradient/6.0;
}


__global__ void d_GradientFilter(const CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_DEPTH_TYPE* d_resultDepthMap, uint width, uint height) {
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= width) || (y >= height)) return;
	const uint offset = width*y + x;
	const float threshold_squared = 100.f; // 10^2

	const float2 gradient = d_ComputeGradientByPrewitt(d_newDepthMap, x, y, width, height);
	CAMERA_DEPTH_TYPE newValue = d_newDepthMap[offset];
	if (gradient.x * gradient.x + gradient.y * gradient.y > threshold_squared) {
		newValue = 0.f;
	}

	d_resultDepthMap[offset] = newValue;
	
}

__device__
CAMERA_DEPTH_TYPE getMedian(int n, CAMERA_DEPTH_TYPE x[]) {
    CAMERA_DEPTH_TYPE temp;
    int i, j;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }
 
    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        return((x[n/2] + x[n/2 - 1]) / 2.0);
    } else {
        // else return the element in the middle
        return x[n/2];
    }
}

__global__ void d_MedianFilter(const CAMERA_DEPTH_TYPE* d_newDepthMap, CAMERA_DEPTH_TYPE* d_resultDepthMap, uint width, uint height,CAMERA_DEPTH_TYPE depthNullValue) {
	const uint x = blockIdx.x*blockDim.x + threadIdx.x;
    const uint y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((x >= width) || (y >= height)) return;
	const uint offset = width*y + x;

	const int arraySize = 25;
	const int medianRadius = 2;
	int counter = 0;
	CAMERA_DEPTH_TYPE tempArr[arraySize] = {0};

	for (int i = -medianRadius ; i <= medianRadius ; ++i) {
		for (int j = -medianRadius ; j <= medianRadius ; ++j) {
			CAMERA_DEPTH_TYPE depth=d_newDepthMap[width*((y + j) % height) + ((x + i) % width)];
			if (depth != depthNullValue) {
				tempArr[counter++] = depth;
			}
		}
	}
	if (counter <= (arraySize/2)) {
		d_resultDepthMap[offset] = depthNullValue;
	} else {
		d_resultDepthMap[offset] = getMedian(counter, tempArr);
	}
}

//////////////////////////////CPP Functions ///////////////////////////////////////////////////////////


void CameraAbstract::AverageFilter(CAMERA_DEPTH_TYPE* d_newDepthMap) {
	dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
	// TODO - MAKE THIS GRID_DIM A PARAMETER OF CAMERA_ABSTRACT. IT SHOULD BE THE RESOLUTION OF THE PICTURE. UPDATE_CAMERA_DATA SHOULD USE IT TOO
	d_AverageFilter<<<gridSize, blockSize>>>(d_newDepthMap, m_tempDepthArray, m_params.m_width, m_params.m_height, m_params.depthNullValue);
	cutilSafeCall(cudaDeviceSynchronize());
	cutilSafeCall(cudaMemcpy(d_newDepthMap, m_tempDepthArray, m_depthSizeBytes, cudaMemcpyDeviceToDevice));
	cutilSafeCall(cudaDeviceSynchronize());

	//TODO - fix this so it is a real average filter, and not just proximal
}

void CameraAbstract::GradientFilter(CAMERA_DEPTH_TYPE* d_newDepthMap){
	dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
	d_GradientFilter<<<gridSize, blockSize>>>(d_newDepthMap, m_tempDepthArray, m_params.m_width, m_params.m_height);
	cutilSafeCall(cudaDeviceSynchronize());
	cutilSafeCall(cudaMemcpy(d_newDepthMap, m_tempDepthArray, m_depthSizeBytes, cudaMemcpyDeviceToDevice));
	cutilSafeCall(cudaDeviceSynchronize());
}

void CameraAbstract::MedianFilter(CAMERA_DEPTH_TYPE* d_newDepthMap){
	dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
	d_MedianFilter<<<gridSize, blockSize>>>(d_newDepthMap, m_tempDepthArray, m_params.m_width, m_params.m_height,m_params.depthNullValue);
	cutilSafeCall(cudaDeviceSynchronize());
	cutilSafeCall(cudaMemcpy(d_newDepthMap, m_tempDepthArray, m_depthSizeBytes, cudaMemcpyDeviceToDevice));
	cutilSafeCall(cudaDeviceSynchronize());
}


#define	FRAC(x) (x-floor(x))
#define INTERP(f1,f2,x) ((1-x)*f1+(x)*f2)

__global__ void d_undistort(CameraParams m_params, CAMERA_DEPTH_TYPE* depthMap,
							CAMERA_DEPTH_TYPE* newDepthMap )
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint& width = m_params.m_width;
	uint& height = m_params.m_height;
	if ((x >= width) || (y >= height)) return;

	float fx = m_params.m_intrinsic.m_fx;
	float fy = m_params.m_intrinsic.m_fy;
	float cx = m_params.m_intrinsic.m_cx;
	float cy = m_params.m_intrinsic.m_cy;
	float k1 = m_params.k1;
	float k2 = m_params.k2;
	float p1 = m_params.p1;
	float p2 = m_params.p2;
	
	float xd=(x-cx)/fx;
	float yd=(y-cy)/fy;
	
	float r2=xd*xd+yd*yd;
	xd=xd*(1+k1*r2 + k2*r2*r2) + 2*p1*xd*yd + p2*(r2 + 2*xd*xd);
	yd=yd*(1+k1*r2 + k2*r2*r2) + 2*p2*xd*yd + p1*(r2 + 2*yd*yd);

	float u = fx*xd + cx;
	float v = fy*yd + cy;
	
	
	if (floor(u)>width-1 || floor(u)<0 || floor(v)<0 || floor(v)>height-1) {
		return;	//Todo: handle edges
	}

	CAMERA_DEPTH_TYPE f1,f2;
	int offset;
	//Bottom right pixle case
	if (floor(u)==width-1 && floor(v)==height-1) {
		offset=floor(v)*width+floor(u);
		f1=f2=depthMap[offset];
	}
	//Right edge case
	else if (floor(u) == width-1) {
		offset = floor(v)*width+floor(u);
		f1 = depthMap[offset];
		offset = (floor(v)+1)*width+floor(u);
		f2 = depthMap[offset];
	}
	else {
		offset=floor(v)*width+floor(u);
		f1= INTERP(depthMap[offset],depthMap[offset+1],FRAC(u));
		//Bottom edge case
		if (floor(u) == height-1) {
			f2=f1;
		} else {
			offset=(floor(v)+1)*width+floor(u);
			f2= INTERP(depthMap[offset],depthMap[offset+1],FRAC(u));
		}	
	}
	offset=y*width+x;
	newDepthMap[offset]=INTERP(f1,f2,FRAC(v));
	return;
}



void CameraAbstract::undistort(CameraParams m_params, 
	CAMERA_DEPTH_TYPE* depthMap,  CAMERA_DEPTH_TYPE* newDepthMap ) {
	dim3 gridSize(iDivUp(m_params.m_width, blockSize.x), iDivUp(m_params.m_height, blockSize.y));
	d_undistort<<<gridSize, blockSize>>>(m_params,depthMap,newDepthMap);
}