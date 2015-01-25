
#include "ICP.h"
#include "CameraHandler.h"
#include "utilities.h"

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA Includes
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

#include <iostream> // TODO REMOVE
#include <fstream> // TODO REMOVE

#include <math.h>

using namespace std; // TODO REMOVE

#include <Eigen/Core>
#include <Eigen/Dense>

static Eigen::Matrix4f m_identity4f; // TODO: PUT IN CLASS. SOMETHING IS WRONG WITH THE INCLUDES OF EIGEN IN THE .H FILE

ICP::ICP (dim3 gridSize, dim3 blockSize, CameraParams* cameraParams) : 
	m_gridSize(gridSize), m_blockSize(blockSize), m_maxIterations(ICP_ITERATIONS), m_cameraParams(*cameraParams)
{
	uint width = m_cameraParams.m_width;
	uint height = m_cameraParams.m_height;

	m_numCorrespondenceBlocks = iDivUp(width, BLOCK_SIZE)*iDivUp(height, BLOCK_SIZE);

	//allocate memory in cuda, needs to be done only once, later we will just change the values.
	cutilSafeCall(cudaMalloc((void**)&d_A_Matrix, width*height*NUM_VARS*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&d_b_Matrix, width*height*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&AtA, NUM_VARS*NUM_VARS*m_numCorrespondenceBlocks*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&Atb, NUM_VARS*m_numCorrespondenceBlocks*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&invLastTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&newTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&incrementalTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&bestTransform, 4*4*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&Atransposed, width*height*NUM_VARS*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&Atransposed_x_A, NUM_VARS*NUM_VARS*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&xOpt, NUM_VARS*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&Atransposed_x_b, NUM_VARS*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&inv_Atransposed_x_A, NUM_VARS*NUM_VARS*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&errorSum, sizeof(float)));

	Atransposed_x_A_host = new float[NUM_VARS*NUM_VARS];
	invLastTransform_host = new float[4*4];

	AtA_host = new float[NUM_VARS*NUM_VARS*m_numCorrespondenceBlocks];
	Atb_host = new float[NUM_VARS*m_numCorrespondenceBlocks];
	xOpt_host = new float[NUM_VARS];

	m_identity4f.setIdentity();
}

ICP::~ICP()
{
	cutilSafeCall(cudaFree(d_A_Matrix));
	cutilSafeCall(cudaFree(d_b_Matrix));
	cutilSafeCall(cudaFree(AtA));
	cutilSafeCall(cudaFree(Atb));
	cutilSafeCall(cudaFree(invLastTransform));
	cutilSafeCall(cudaFree(newTransform));
	cutilSafeCall(cudaFree(incrementalTransform));
	cutilSafeCall(cudaFree(bestTransform));
	cutilSafeCall(cudaFree(Atransposed_x_A));
	cutilSafeCall(cudaFree(inv_Atransposed_x_A));
	cutilSafeCall(cudaFree(xOpt));
	cutilSafeCall(cudaFree(Atransposed_x_b));
	cutilSafeCall(cudaFree(errorSum));

	delete[] Atransposed_x_A_host;
	delete[] invLastTransform_host;
	delete[] xOpt_host;
	delete[] AtA_host;
	delete[] Atb_host;

}

void ICP::Iterate_ICP(const CameraData& d_newDepthData,
					  const PointInfo& d_renderedVertexMap,
					  float* d_currentTransform)	// This is input & output parameter on cuda
{
	//static const Eigen::Matrix4f m_identity4f = Eigen::Matrix4f::Identity();
//	Eigen::Matrix4f m_identity4f;
//	m_identity4f.setIdentity();
	cutilSafeCall( cudaMemcpy(incrementalTransform, m_identity4f.data(), 16*sizeof(float), cudaMemcpyHostToDevice) );
	
	// d_currentTransform currently holds the previous transformation matrix. Initialize new transformation matrix to the previous
	cutilSafeCall( cudaMemcpy(newTransform, d_currentTransform, 16*sizeof(float), cudaMemcpyDeviceToDevice) );

	int iterationCounter = 0;

	cutilSafeCall(cudaMemcpy(bestTransform, newTransform, 16*sizeof(float), cudaMemcpyDeviceToDevice));

	// While loop - until we get minimum transformation matrix:
	do {
		iterationCounter++;

		// Copy current transform to its matching field.

		cutilSafeCall( cudaMemcpy(invLastTransform_host, newTransform, 16*sizeof(float), cudaMemcpyDeviceToHost) );
		//cutilSafeCall( cudaMemcpy(invLastTransformMatrix.data(), newTransform, 16*sizeof(float), cudaMemcpyDeviceToHost) );
		cudaDeviceSynchronize();
		// Calculate inverse of current transformation matrix and store in invLastTransformMatrix.
		Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor> > invLastTransformMatrix(invLastTransform_host);	// TODO: COLMAJOR IS ALSO WORKING WELL. CHECK OUT. MAYBE BUGGY
		invLastTransformMatrix = invLastTransformMatrix.inverse();
		
		// Copy current inverse transformation matrix into cuda.
		cutilSafeCall( cudaMemcpy(invLastTransform, invLastTransform_host, 16*sizeof(float), cudaMemcpyHostToDevice) );
		cudaDeviceSynchronize();

		// Find corresponding points and build A and b matrices.
		FindCorresponding (d_newDepthData, d_renderedVertexMap);

		//////calculating Xopt = (At * A)^(-1) * At * b  using tree reduction//////

		cutilSafeCall(cudaDeviceSynchronize());
		ClearHostMatrices();

		cutilSafeCall(cudaMemcpy(AtA_host, AtA, NUM_VARS*NUM_VARS*m_numCorrespondenceBlocks*sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(Atb_host, Atb, NUM_VARS*m_numCorrespondenceBlocks*sizeof(float), cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < m_numCorrespondenceBlocks ; ++i) {
			for (int j = 0; j < NUM_VARS*NUM_VARS; ++j) {
				AtA_sum[j] += AtA_host[i * NUM_VARS*NUM_VARS + j];
			}
			for (int j = 0; j < NUM_VARS; ++j) {
				Atb_sum[j] += Atb_host[i * NUM_VARS + j];
			}
		}
		for (int i = 0; i < NUM_VARS; ++i) {
			for (int j = 0; j < i; ++j) {
				AtA_sum[i * NUM_VARS + j] = AtA_sum[j * NUM_VARS + i];
			}
		}

		Eigen::Matrix<float, NUM_VARS, NUM_VARS, Eigen::RowMajor> AtA_eigen;
		Eigen::Matrix<float, NUM_VARS, 1> Atb_eigen;

		for (int i = 0; i < NUM_VARS; ++i) {
			Atb_eigen(i) = Atb_sum[i];
			for (int j = 0; j < NUM_VARS; ++j) {
				AtA_eigen(i, j) = AtA_sum[i * NUM_VARS + j];
			}
		}

		float det = AtA_eigen.determinant();
		if (isnan(det) || det == 0.f || fabs(det) < _EPSILON_) {
			// TODO - PROBLEM! MATRIX IS SINGULAR. HANDLE
			cutilSafeCall( cudaMemcpy(newTransform, d_currentTransform, 16*sizeof(float), cudaMemcpyDeviceToDevice) );
			cout << "       No transform found." << endl;
			break;
		}

		Eigen::Matrix<float, NUM_VARS, 1> parameters = AtA_eigen.llt().solve(Atb_eigen).cast<float>();
		
		for (int i = 0; i < 3; ++i) {
			xOpt_host[i] = -parameters(i);		// TODO - CONSIDER WRITE -parameters(i) (minus) like in KinectShape
		}

		for (int i = 3; i < 6; ++i) {	// Angles are negated, translations are positive
			xOpt_host[i] = parameters(i);
		}

		cutilSafeCall(cudaMemcpy(xOpt, xOpt_host, NUM_VARS*sizeof(float), cudaMemcpyHostToDevice));

		BuildNewTransformMatrix();

		CameraHandler::updateCameraData(m_gridSize, m_blockSize, d_newDepthData, NULL, newTransform, NULL, NULL);
		cutilSafeCall(cudaDeviceSynchronize());

	} while (iterationCounter < m_maxIterations);
		
	cutilSafeCall(cudaMemcpy(d_currentTransform, newTransform, 16*sizeof(float), cudaMemcpyDeviceToDevice));
	
	// The output is in @Param d_currentTransform
}

void ICP::ClearHostMatrices(){
	// Initialize AtA_sum and Atb_sum
	for (int i = 0; i < NUM_VARS*NUM_VARS; ++i) {
		AtA_sum[i] = 0.f;
	}
	for (int i = 0; i < NUM_VARS; ++i) {
		Atb_sum[i] = 0.f;
	}

	//TODO - is there any point in initializing xOpt_host, AtA_host, Atb_host?  They were allocated each iteration, but never cleared
}





