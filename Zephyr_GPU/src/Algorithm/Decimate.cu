#include "Decimate.h"
#include "Quadric_GPU.cuh"
#include "QEM_Data.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/mutex.h>
#include <map>
#include <thread>

#include <Timer.h>
#include <Random.h>

#include "../Random.cuh"
#include "../QueryDevice.h"

#include <Decimate/QuadricError.h>

using namespace Zephyr;
using namespace Zephyr::Common;
using namespace Zephyr::GPU;

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

__constant__ double MAX_ERRORS[2]; // quadric error, max flip error

/*
__device__
double computeFlipAngle(INDEX_TYPE halfEdgeId, INDEX_TYPE* index, Vector3f* vertices)
{
	// Set the maximum angular deviation of the orignal normal and the new normal in degrees.
	double max_deviation_ = maxAngle / 180.0 * M_PI;
	double min_cos_ = cos(max_deviation_);

	// check for flipping normals
	OMMesh::ConstVertexFaceIter vf_it(omesh, collapseInfo.v0);
	FaceHandle					fh;
	OMMesh::Scalar              c(1.0);

	// put point to remain in vertex to be removed
	omesh.set_point(collapseInfo.v0, collapseInfo.p1);

	for (; vf_it.is_valid(); ++vf_it)
	{
		fh = *vf_it;
		if (fh != collapseInfo.fl && fh != collapseInfo.fr)
		{
			OMMesh::Normal n1 = omesh.normal(fh);
			OMMesh::Normal n2 = omesh.calc_face_normal(fh);

			c = dot(n1, n2);

			if (c < min_cos_)
				break;
		}
	}

	// undo simulation changes
	omesh.set_point(collapseInfo.v0, collapseInfo.p0);

	return float((c < min_cos_) ? INVALID_COLLAPSE : c);
}
*/
__device__
Quadric_GPU computeFaceQE(INDEX_TYPE IdStart, INDEX_TYPE* index, Vector3f* vertices)
{
	Vector3f v0 = vertices[index[IdStart]];
	Vector3f v1 = vertices[index[IdStart + 1]];
	Vector3f v2 = vertices[index[IdStart + 2]];

	Vector3f n = (v1 - v0).cross(v2 - v0);

	double area = n.norm();

	if (area > FLT_MIN)
	{
		n /= area;
		area *= 0.5;
	}

	double a = n[0];
	double b = n[1];
	double c = n[2];
	double d = -(v0.dot(n));

	Quadric_GPU q(a, b, c, d);
	q *= area;

	return q;
}

__device__
double computeError(int id, QEM_Data* QEM_Datas)
{
	auto data = QEM_Datas[id];
	
	// if invalid just return max error
	if (!data.bValid)
		return 10000.0;

	Quadric_GPU q;
	for (int i = 0; i < data.indexCount; i += 3)
	{
		q += computeFaceQE(i, data.indices, data.vertices);
	}

	double err = q.evalute(data.vertices[data.vertexToKeepId]);

	//printf("err: %f\n", err);

	return (err < MAX_ERRORS[0]) ? err : 10000.0;
}

__global__
void computeErrors(QEM_Data* QEM_Datas, double* errors)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	errors[i] = computeError(i, QEM_Datas);
}

__global__
void selectBestEdge(double* errors, int* random, int* bestEdge)
{
	extern __shared__ double sErrors[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// copy the errors to shared memory
	sErrors[threadIdx.x] = errors[i];
	__syncthreads();

	// Only one thread per block can search for the best edge
	if (0 == threadIdx.x)
	{
		double bestError = 10000.0f;
		int bestHalfEdge = -1;
		for (int i = 0; i < blockDim.x; ++i)
		{
			double err = sErrors[i];
			if (bestError > err)
			{
				bestError = err;
				bestHalfEdge = random[blockIdx.x * blockDim.x + i];
			}
		}
		//printf("id: %d, Score: %f", bestHalfEdge, bestError);
		bestEdge[blockIdx.x] = bestHalfEdge;
	}
}

struct ConstError
{
	ConstError(double maxQuadricError_, double maxNormalFlipDeviation_) 
		: maxQuadricError(maxQuadricError_), maxNormalFlipDeviation(maxNormalFlipDeviation_) {}
	
	double maxQuadricError;
	double maxNormalFlipDeviation;
};

int GPU::decimate(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	const float maxQuadricError = 0.1f;
	const float maxNormalFlipDeviation = 45.0;
	const int maxRetryCount = 50;

	// set the constant memory data
	ConstError constError(maxQuadricError, maxNormalFlipDeviation);
	cudaMemcpyToSymbol(MAX_ERRORS, (ConstError*)&constError, sizeof(constError));

	auto& omesh = mesh.getMesh();

	int retryCount = 0;
	size_t initialFaceCount = omesh.n_faces();
	size_t currentFaceCount = initialFaceCount;
	size_t totalHalfEdgeCount = omesh.n_halfedges();
	size_t totalCollapseRequired = (initialFaceCount - targetFaceCount) / 2;

	int numOfThreads = std::thread::hardware_concurrency();
	std::vector<std::vector<std::pair<float, HalfedgeHandle>>> selectedErrorEdgesPerThread(numOfThreads);
	std::vector<std::shared_ptr<RandomGenerator>> randomGenerators;

	//QueryDevice::printQuery();

	// compute the total number of block we require to complete the task
	int N = totalCollapseRequired;
	int threadPerBlock = binSize;

	// check how many block we can run together at once
	int oneIterationBlockCount = QueryDevice::computeOptimalBlockCount(N, threadPerBlock, 0);
	int oneIterationSelectionSize = oneIterationBlockCount * threadPerBlock;
	std::cout << "1 iteration block count: " << oneIterationBlockCount << std::endl;
	std::cout << "1 iteration selection size: " << oneIterationSelectionSize << std::endl;

	int randomSequence = 0;
	thrust::device_vector<int> d_randomEdgesId(oneIterationSelectionSize);
	thrust::host_vector<int, thrust::system::cuda::experimental::pinned_allocator<int>> h_randomEdgesId(oneIterationSelectionSize);
	thrust::device_vector<double> d_Errors(oneIterationBlockCount);
	thrust::device_vector<int> d_bestEdges(oneIterationBlockCount);
	thrust::host_vector <int, thrust::system::cuda::experimental::pinned_allocator<int>> h_BestEdge(oneIterationBlockCount);

	CopyPartialMesh partialMesh(oneIterationSelectionSize);

	int* d_BestEdges_ptr = thrust::raw_pointer_cast(&d_bestEdges[0]);
	int* d_randomEdgesId_ptr = thrust::raw_pointer_cast(&d_randomEdgesId[0]);
	double* d_Errors_ptr = thrust::raw_pointer_cast(&d_Errors[0]);
	
	int totalCollapseCount = 0;
	// do the exact collapse
	int collapseCount = std::numeric_limits<int>::max();
	while (retryCount < maxRetryCount && currentFaceCount > targetFaceCount && collapseCount > (0.05 * oneIterationBlockCount))
	{
		collapseCount = 0;
		Timer time;
		Random::generateRandomInt(d_randomEdgesId, 0, (int)totalHalfEdgeCount - 1, randomSequence);
		h_randomEdgesId = d_randomEdgesId;
		// advance the randomSequence
		randomSequence += N;
		//std::cout << "Generate Random time: " << time.getElapsedTime() << std::endl;
		
		// Data marshalling and CUDA kernel call
		{
			Timer copyPartialTimer;
			QEM_Data* d_QEM_Datas_ptr = partialMesh.copyPartialMesh(omesh, h_randomEdgesId);
			//std::cout << "Copy partial time: " << copyPartialTimer.getElapsedTime() << std::endl;

			Timer QEM_Package_Timer;
			//std::cout << "QEM_Package time: " << QEM_Package_Timer.getElapsedTime() << std::endl;

			Timer ComputeErrorTimer;
			// Compute the Error of each random edge selected.
			computeErrors <<< oneIterationBlockCount, threadPerBlock >>>(d_QEM_Datas_ptr, d_Errors_ptr);
			//std::cout << "Compute Error time: " << ComputeErrorTimer.getElapsedTime() << std::endl;
			CudaCheckError();
			// Compare and find the edge with best score.

			Timer bestEdgeTimer;
			selectBestEdge <<< oneIterationBlockCount, threadPerBlock, threadPerBlock * sizeof(double) >>>(d_Errors_ptr, d_randomEdgesId_ptr, d_BestEdges_ptr);
			//std::cout << "Best Edge time: " << bestEdgeTimer.getElapsedTime() << std::endl;
			CudaCheckError();
			// copy the result of the kernel back to host
			h_BestEdge = d_bestEdges;
			//std::cout << "Exiting" << std::endl;
		}
		
		Timer collapseTimer;

		int faceCollapsed = 0;
		for (auto bestEdgeId : h_BestEdge)
		{
			if (bestEdgeId == -1)
				continue;

			HalfedgeHandle halfEdgeHandle(bestEdgeId);
			if (!omesh.is_collapse_ok(halfEdgeHandle))
				continue;

			// if the edge is a boundary edge only 1 face is removed by the collapse, otherwise 2 face is removed
			faceCollapsed += omesh.is_boundary(halfEdgeHandle) ? 1 : 2;

			omesh.collapse(halfEdgeHandle);
			++collapseCount;
		}
		totalCollapseCount += collapseCount;

		//std::cout << "Total Collapsed this iteration: " << collapseCount << " Total Collapsed: " << totalCollapseCount  << std::endl;
		//std::cout << "Original Collapsed this iteration: " << originalCollapse << std::endl;
		//std::cout << "Additional Collapsed this iteration: " << additionalCollapse << std::endl;
		//std::cout << "Total Collapsed : " << totalCollapseCount << std::endl;

		// if there is no changes in face count, retry
		if (0 == collapseCount)
		{
			++retryCount;
			std::cout << "Retrying: " << retryCount << std::endl;
		}

		currentFaceCount -= faceCollapsed;

		//std::cout << "Collapse time: " << collapseTimer.getElapsedTime() << std::endl;
	}
	omesh.garbage_collection();

	if (retryCount == maxRetryCount)
	{
		std::cout << "Reach max retry count of " << maxRetryCount << ", teriminating early" << std::endl;
	}

	return totalCollapseCount;
}