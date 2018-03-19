#include "Decimate.h"

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

__global__
void selectBestEdge(int n, float a)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
}

int GPU::decimate(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	const float maxQuadricError = 0.01f;
	const float maxNormalFlipDeviation = 15.0f;
	const int maxRetryCount = 500;

	auto& omesh = mesh.getMesh();

	int retryCount = 0;
	size_t initialFaceCount = omesh.n_faces();
	size_t currentFaceCount = initialFaceCount;
	size_t totalHalfEdgeCount = omesh.n_halfedges();
	size_t totalCollapseRequired = (initialFaceCount - targetFaceCount) / 2;

	int numOfThreads = std::thread::hardware_concurrency();
	std::vector<std::vector<std::pair<float, HalfedgeHandle>>> selectedErrorEdgesPerThread(numOfThreads);
	std::vector<std::shared_ptr<RandomGenerator>> randomGenerators;

	QueryDevice::printQuery();

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
	thrust::device_vector<int> randomEdgesId(oneIterationSelectionSize);

	int totalCollapseCount = 0;
	while (retryCount < maxRetryCount && currentFaceCount > targetFaceCount)
	{
		Random::generateRandomInt(randomEdgesId, 0, (int)totalHalfEdgeCount - 1, randomSequence);
		// advance the randomSequence
		randomSequence += N;
	}
	omesh.garbage_collection();

	if (retryCount == maxRetryCount)
	{
		std::cout << "Reach max retry count of " << maxRetryCount << ", teriminating early" << std::endl;
	}

	return totalCollapseCount;
}