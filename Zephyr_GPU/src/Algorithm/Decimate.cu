#include "Decimate.h"
#include "Quadric_GPU.cuh"
#include "OpenMesh_GPU.h"

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

__constant__ double MAX_ERRORS[2]; // quadric error, max flip error

__device__
Quadric_GPU computeFaceQE(INDEX_TYPE IdStart, INDEX_TYPE* index, Vector3f* vertices)
{
	Vector3f v0 = vertices[index[IdStart]];
	Vector3f v1 = vertices[index[IdStart + 1]];
	Vector3f v2 = vertices[index[IdStart + 2]];

	//printf("V0: %f, %f, %f\n", v0.x(), v0.y(), v0.z());
	//printf("V1: %f, %f, %f\n", v1.x(), v1.y(), v1.z());
	//printf("V2: %f, %f, %f\n", v2.x(), v2.y(), v2.z());

	Vector3f n = (v1 - v0).cross(v2 - v0);

	//printf("Normal: %f, %f, %f\n", n.x(), n.y(), n.z());

	double area = n.norm();

	//printf("Area: %f\n", area);

	if (area > FLT_MIN)
	{
		n /= area;
		area *= 0.5;
	}

	double a = n[0];
	double b = n[1];
	double c = n[2];
	double d = -(v0.dot(n));

	//printf("%d: %f, %f, %f, %f\n", IdStart, a, b, c, d);

	Quadric_GPU q(a, b, c, d);
	q *= area;

	return q;
}

__device__
double computeQE(int id, QEM_Data_GPU* QEM_Datas)
{
	auto data = QEM_Datas[id];
	
	Quadric_GPU q;
	for (int i = 0; i < data.indexCount; i += 3)
	{
		q += computeFaceQE(i, data.indices, data.vertices);
	}

	double err = q.evalute(data.vertices[data.vertexToKeepId]);

	return (err < MAX_ERRORS[0]) ? err : 10000.0;
}

__global__
void selectBestEdge(int* randomNum, int* bestEdges, QEM_Data_GPU* QEM_Datas)
{
	extern __shared__ float s[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	bestEdges[i] = randomNum[i];

	double QEM = computeQE(i, QEM_Datas);

	s[threadIdx.x] = QEM;
	printf("%f\n", QEM);
	bestEdges[i] = s[threadIdx.x];
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
	const float maxQuadricError = 0.01f;
	const float maxNormalFlipDeviation = 15.0f;
	const int maxRetryCount = 500;

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
	int numOfBlock = (N / threadPerBlock) + 1;

	// check how many block we can run together at once
	int oneIterationBlockCount = QueryDevice::computeOptimalBlockCount(N, threadPerBlock, 0);
	int oneIterationSelectionSize = oneIterationBlockCount * threadPerBlock;
	std::cout << "1 iteration block count: " << oneIterationBlockCount << std::endl;
	std::cout << "1 iteration selection size: " << oneIterationSelectionSize << std::endl;

	int randomSequence = 0;
	thrust::device_vector<int> d_randomEdgesId(oneIterationSelectionSize);
	thrust::device_vector<int> d_bestEdges(oneIterationBlockCount);
	thrust::host_vector<int> h_BestEdge(oneIterationBlockCount);

	int* d_BestEdges_ptr = thrust::raw_pointer_cast(&d_bestEdges[0]);
	int* d_randomEdgesId_ptr = thrust::raw_pointer_cast(&d_randomEdgesId[0]);
	
	int totalCollapseCount = 0;
	while (retryCount < maxRetryCount && currentFaceCount > targetFaceCount)
	{
		Random::generateRandomInt(d_randomEdgesId, 0, (int)totalHalfEdgeCount - 1, randomSequence);
		// advance the randomSequence
		randomSequence += N;

		// Data marshalling 
		OpenMesh_GPU gpu_mesh;

		std::vector<int> randomEdgesId(d_randomEdgesId.begin(), d_randomEdgesId.end());
		auto& QEM_Datas = gpu_mesh.copyPartialMesh(omesh, randomEdgesId);

		{
			QEM_Data_Package QEM_package(QEM_Datas);
			QEM_Data_GPU* d_QEM_Datas_ptr = QEM_package.mp_QEM_Data_GPU;

			// Call the best edge kernel
			selectBestEdge <<< oneIterationBlockCount, threadPerBlock, threadPerBlock*sizeof(int) >>>(d_randomEdgesId_ptr, d_BestEdges_ptr, d_QEM_Datas_ptr);
		}
		// copy the result of the kernel back to host
		h_BestEdge = d_bestEdges;

		// do the exact collapse
		int collapseCount = 0;
		int originalCollapse = 0;
		int additionalCollapse = 0;

		int faceCollapsed = 0;
		int threadCount = -1;
		for (auto bestEdgeId : h_BestEdge)
		{
			std::cout << bestEdgeId << std::endl;
			
			++threadCount;
			if (bestEdgeId == -1)
				continue;

			HalfedgeHandle halfEdgeHandle(bestEdgeId);
			if (!omesh.is_collapse_ok(halfEdgeHandle))
				continue;

			// if the edge is a boundary edge only 1 face is removed by the collapse, otherwise 2 face is removed
			faceCollapsed += omesh.is_boundary(halfEdgeHandle) ? 1 : 2;

			omesh.collapse(halfEdgeHandle);
			++collapseCount;
			++originalCollapse;
		}
		totalCollapseCount += collapseCount;

		//std::cout << "Total Collapsed this iteration: " << collapseCount << std::endl;
		//std::cout << "Original Collapsed this iteration: " << originalCollapse << std::endl;
		//std::cout << "Additional Collapsed this iteration: " << additionalCollapse << std::endl;
		//std::cout << "Total Collapsed : " << totalCollapseCount << std::endl;

		// if there is no changes in face count, retry
		if (0 == collapseCount)
		{
			++retryCount;
			//std::cout << "Retrying: " << retryCount << std::endl;
		}

		currentFaceCount -= faceCollapsed;
	}
	omesh.garbage_collection();

	if (retryCount == maxRetryCount)
	{
		std::cout << "Reach max retry count of " << maxRetryCount << ", teriminating early" << std::endl;
	}

	return totalCollapseCount;
}