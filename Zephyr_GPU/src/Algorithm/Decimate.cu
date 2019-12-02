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
#define CUDA_ERROR_CHECK

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

__constant__ float MAX_ERRORS[2]; // quadric error, max flip error

__device__
Quadric_GPU computeFaceQE(INDEX_TYPE IdStart, INDEX_TYPE* index, Vector3f* vertices)
{
	Vector3f v0 = vertices[index[IdStart]];
	Vector3f v1 = vertices[index[IdStart + 1]];
	Vector3f v2 = vertices[index[IdStart + 2]];

	Vector3f n = (v1 - v0).cross(v2 - v0);

	float area = n.norm();

	if (area > FLT_MIN)
	{
		n /= area;
		area *= 0.5;
	}

	float a = n[0];
	float b = n[1];
	float c = n[2];
	float d = -(v0.dot(n));

	Quadric_GPU q(a, b, c, d);
	q *= area;

	return q;
}

__device__
float computeError(int id, QEM_Data* QEM_Datas)
{
	auto data = QEM_Datas[id];
	
	// if invalid just return max error
	if (!data.bValid)
		return 10000.0f;

	Quadric_GPU q;
	for (int i = 0; i < data.indexCount; i += 3)
	{
		q += computeFaceQE(i, data.indices, data.vertices);
	}

	float err = q.evalute(data.vertices[data.vertexToKeepId]);

	//printf("err: %f\n", err);

	return (err < MAX_ERRORS[0]) ? err : 10000.0f;
}

__global__
void computeErrors(QEM_Data* QEM_Datas, float* errors)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	errors[i] = computeError(i, QEM_Datas);
}

__global__
void selectBestEdge(float* errors, int* random, int* bestEdge)
{
	extern __shared__ float sErrors[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// copy the errors to shared memory
	sErrors[threadIdx.x] = errors[i];
	__syncthreads();

	// Only one thread per block can search for the best edge
	if (0 == threadIdx.x)
	{
		float bestError = 10000.0f;
		int bestHalfEdge = -1;
		for (int i = 0; i < blockDim.x; ++i)
		{
			float err = sErrors[i];
			if (err < MAX_ERRORS[0] && err < bestError)
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
	ConstError(float maxQuadricError_, float maxNormalFlipDeviation_)
		: maxQuadricError(maxQuadricError_), maxNormalFlipDeviation(maxNormalFlipDeviation_) {}
	
	float maxQuadricError;
	float maxNormalFlipDeviation;
};

int GPU::decimate(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize, Algorithm::DecimationType type)
{
	Common::Timer timer;

	auto& omesh = mesh.getMesh();

	int collapseCount = -1;
	auto previousFaceCount = omesh.n_faces();

	std::cout << "Using ";
	if (Algorithm::DecimationType::GPU_RANDOM_DECIMATE == type)
	{
		std::cout << "GPU Random Decimation..." << std::endl;
		collapseCount = GPU::decimateMC(mesh, targetFaceCount, binSize);
	}
	else if (Algorithm::DecimationType::GPU_SUPER_VERTEX == type)
	{
		std::cout << "GPU Super Vertex..." << std::endl;
		collapseCount = GPU::decimateSuperVertex(mesh, targetFaceCount, binSize);
	}

	auto elapseTime = timer.getElapsedTime();
	auto& omeshDecimated = mesh.getMesh();
	omesh.garbage_collection();

	std::cout << "Decimation done in " << elapseTime << " sec" << std::endl;
	std::cout << "Original Face Count: " << previousFaceCount << std::endl;
	std::cout << "Target Face Count: " << targetFaceCount << std::endl;
	std::cout << "Removed Face Count: " << collapseCount << std::endl;
	std::cout << "Decimated Face Count: " << omeshDecimated.n_faces() << std::endl;
	std::cout << "Percentage decimated: " << ((previousFaceCount - omeshDecimated.n_faces()) / (float)previousFaceCount) * 100.0f << " %" << std::endl;

	return collapseCount;
}

int GPU::decimateMC(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	Timer totalTimer("Total Time");

	const float maxQuadricError = 0.01f;
	const float maxNormalFlipDeviation = 15.0f;
	const int maxRetryCount = 250;

	// set the constant memory data
	ConstError constError(maxQuadricError, maxNormalFlipDeviation);
	cudaMemcpyToSymbol(MAX_ERRORS, (ConstError*)&constError, sizeof(constError), 0, cudaMemcpyHostToDevice);

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
	thrust::device_vector<float> d_Errors(oneIterationBlockCount);
	thrust::device_vector<int> d_bestEdges(oneIterationBlockCount);
	thrust::host_vector <int, thrust::system::cuda::experimental::pinned_allocator<int>> h_BestEdge(oneIterationBlockCount);

	CopyPartialMesh partialMesh(oneIterationSelectionSize);

	int* d_BestEdges_ptr = thrust::raw_pointer_cast(&d_bestEdges[0]);
	int* d_randomEdgesId_ptr = thrust::raw_pointer_cast(&d_randomEdgesId[0]);
	float* d_Errors_ptr = thrust::raw_pointer_cast(&d_Errors[0]);
	
	// do the first random generation outside of iteration.
	Random::generateRandomInt(d_randomEdgesId, 0, (int)totalHalfEdgeCount - 1, randomSequence);
	h_randomEdgesId = d_randomEdgesId;

	int totalCollapseCount = 0;
	int garbageCollectTrigger = oneIterationBlockCount / 2;

	double substractTime = 0;

	// do the exact collapse
	int collapseCount = std::numeric_limits<int>::max();
	while (retryCount < maxRetryCount && currentFaceCount > targetFaceCount && collapseCount > (0.05 * oneIterationBlockCount))
	{
		collapseCount = 0;

		// Data marshalling and CUDA kernel call
		{
			// Synchronize the async copy of random edge id from device to host.
			cudaDeviceSynchronize();

			Timer copyPartialTimer;
			QEM_Data* d_QEM_Datas_ptr = partialMesh.copyPartialMesh(omesh, h_randomEdgesId);
			substractTime += copyPartialTimer.getElapsedTime();
			//std::cout << "Copy partial time: " << copyPartialTimer.getElapsedTime() << std::endl;

			//Timer ComputeErrorTimer;
			// Compute the Error of each random edge selected.
			computeErrors <<< oneIterationBlockCount, threadPerBlock >>>(d_QEM_Datas_ptr, d_Errors_ptr);
			//std::cout << "Compute Error time: " << ComputeErrorTimer.getElapsedTime() << std::endl;
			//CudaCheckError();
			// Compare and find the edge with best score.

			//Timer bestEdgeTimer;
			selectBestEdge <<< oneIterationBlockCount, threadPerBlock, threadPerBlock * sizeof(double) >>>(d_Errors_ptr, d_randomEdgesId_ptr, d_BestEdges_ptr);
			//std::cout << "Best Edge time: " << bestEdgeTimer.getElapsedTime() << std::endl;
			//CudaCheckError();
			
			// generate the next set of random number
			//Timer randomTimer;
			Random::generateRandomInt(d_randomEdgesId, 0, (int)totalHalfEdgeCount - 1, randomSequence);
			// Interleave, Async copy from device to host
			cudaMemcpyAsync(thrust::raw_pointer_cast(h_randomEdgesId.data()),
				thrust::raw_pointer_cast(d_randomEdgesId.data()),
				d_randomEdgesId.size()*sizeof(int),
				cudaMemcpyDeviceToHost);
			//std::cout << "Generate Random time: " << randomTimer.getElapsedTime() << std::endl;

			// copy the result of the kernel back to host
			h_BestEdge = d_bestEdges;
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

			CollapseInfo collapseInfo(omesh, halfEdgeHandle);

			if (Algorithm::INVALID_COLLAPSE == Algorithm::QuadricError::computeTriangleFlipAngle(collapseInfo, omesh, maxNormalFlipDeviation) )
				continue;

			// if the edge is a boundary edge only 1 face is removed by the collapse, otherwise 2 face is removed
			faceCollapsed += omesh.is_boundary(halfEdgeHandle) ? 1 : 2;

			omesh.collapse(halfEdgeHandle);
			++collapseCount;
		}

		// Do a garbage collection when invalid cross the trigger threshold
		// Doing garbage collection will remove the deleted edge from the data set, hence removing chance of picking an already deleted edge.
		if (collapseCount < garbageCollectTrigger)
		{
			omesh.garbage_collection();
			totalHalfEdgeCount = omesh.n_halfedges();
			garbageCollectTrigger /= 2;
			//std::cout << "Remaining Half Edge: " << totalHalfEdgeCount << std::endl;
		}

		totalCollapseCount += collapseCount;

		//std::cout << "Total Collapsed this iteration: " << collapseCount << " Total Collapsed: " << totalCollapseCount  << std::endl;
		//std::cout << "Original Collapsed this iteration: " << originalCollapse << std::endl;
		//std::cout << "Additional Collapsed this iteration: " << additionalCollapse << std::endl;
		//std::cout << "Total Collapsed : " << totalCollapseCount << std::endl;
		
		currentFaceCount -= faceCollapsed;

		//std::cout << "Collapse time: " << collapseTimer.getElapsedTime() << std::endl;
	}
	omesh.garbage_collection();

	std::cout << "Substract Time: " << substractTime << std::endl;

	totalTimer.reportTime();
	std::cout << "Substracted Time: " << totalTimer.getElapsedTime() - substractTime << std::endl;

	return totalCollapseCount;
}

// Super Vertex Decimation
struct SV_Header
{
	__device__
	SV_Header() : size(0) {}
	unsigned int size;
};

struct SV_Data
{
	int indexStart[MAX_FACE];
};

__global__
void setupHeaderAndData(int* indices, SV_Header* headers, SV_Data* datas, unsigned char maxFace)
{
	int faceId = blockIdx.x * blockDim.x + threadIdx.x;
	int index = faceId * 3;

	for (int i = 0; i < 3; ++i)
	{
		int vertexId = indices[index + i];

		unsigned size = atomicInc(&headers[vertexId].size, maxFace);
		datas[vertexId].indexStart[size] = index;
	}
}

__device__
Quadric_GPU computeSVFaceQE(int IdStart, int* index, Vector3f* vertices)
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

__global__
void computeSVQuadricAllFace(int* indices, Vector3f* vertices, Quadric_GPU* FaceQuadric)
{
	int faceId = blockIdx.x * blockDim.x + threadIdx.x;
	int index = faceId * 3;

	FaceQuadric[faceId] = computeSVFaceQE(index, indices, vertices);
}

__global__
void computeSVVertexQuadric(SV_Header* headers, SV_Data* datas, Quadric_GPU* faceQuadric, Quadric_GPU* vertexQuadric, size_t vertexCount)
{
	int vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexCount <= vertexId)
		return;
	
	SV_Header header = headers[vertexId];
	SV_Data data = datas[vertexId];

	Quadric_GPU q;
	for (int i = 0; i < header.size; ++i)
	{
		int indexStart = data.indexStart[i];
		int faceId = indexStart / 3;
		q += faceQuadric[faceId];
	}

	vertexQuadric[vertexId] = q;
}

__global__
void selectIndependentVertex(SV_Header* headers, SV_Data* datas, int* indices, bool* vertexUsed, bool* selected, size_t vertexCount)
{
	int vertexId = blockIdx.x * blockDim.x + threadIdx.x;

	if (vertexCount <= vertexId)
		return;
	
	SV_Header header = headers[vertexId];
	SV_Data data = datas[vertexId];

	bool bIndependent = true;
	if (false == vertexUsed[vertexId])
	{
		for (int i = 0; i < header.size && bIndependent; ++i)
		{
			int indexStart = data.indexStart[i];
			for (int j = 0; j < 3 && bIndependent; ++j)
			{
				int checkVertexId = indices[indexStart + j];
				if (true == vertexUsed[checkVertexId])
				{
					bIndependent = false;
				}
			}
		}

		if (bIndependent)
		{
			selected[vertexId] = true;
			vertexUsed[vertexId] = true;
			for (int i = 0; i < header.size; ++i)
			{
				int indexStart = data.indexStart[i];
				for (int j = 0; j < 3; ++j)
				{
					int checkVertexId = indices[indexStart + j];
					vertexUsed[checkVertexId] = true;
				}
			}
		}
	}
}

__global__
void checkIndependentVertex(SV_Header* headers, SV_Data* datas, int* indices, bool* selected, bool* checked, size_t vertexCount)
{
	int vertexId = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertexCount <= vertexId || false == selected[vertexId])
		return;

	SV_Header header = headers[vertexId];
	SV_Data data = datas[vertexId];

	bool bValid = true;
	for (size_t i = 0; i < vertexCount && bValid; ++i)
	{
		if (false == selected[i] && i == vertexId)
			continue;
		
		// check if there is any conflict
		for (int j = 0; j < header.size && bValid; ++j)
		{
			int indexStart = data.indexStart[j];
			for (int k = 0; k < 3 && bValid; ++k)
			{
				int checkVertexId = indices[indexStart + k];
				if (i == checkVertexId) // conflict detected
				{
					bValid = false;
				}
			}
		}
	}
	
	if (bValid)
	{
		// remove self as independent vertex
		checked[vertexId] = true;
	}
}

struct BestEdge
{
	__device__
	BestEdge() : error(10000.0), vertexId(-1) {}
	double error;
	int vertexId;
};

__global__
void getBestEdge(SV_Header* headers, SV_Data* datas, int* indices, Vector3f* vertices, bool* selected, Quadric_GPU* vertexQuadric, BestEdge* BestEdges, size_t vertexCount)
{
	int vertexId = blockIdx.x * blockDim.x + threadIdx.x;

	if (vertexCount <= vertexId || false == selected[vertexId])
		return;
	
	SV_Header header = headers[vertexId];
	SV_Data data = datas[vertexId];
	Quadric_GPU q = vertexQuadric[vertexId];
	
	double bestError = 10000.0;
	int bestVertex = -1;
	for (int i = 0; i < header.size; ++i)
	{
		int indexStart = data.indexStart[i];
		for (int j = 0; j < 3; ++j)
		{
			int checkVertexId = indices[indexStart + j];
			double error = q.evalute(vertices[checkVertexId]);
			if (error < bestError)
			{
				bestError = error;
				bestVertex = checkVertexId;
			}
		}
	}

	BestEdges[vertexId].vertexId = bestVertex;
	BestEdges[vertexId].error = bestError;
}

__global__
void sortBestEdge()
{

}

__global__
void collapse()
{

}

int ZEPHYR_GPU_API Zephyr::GPU::decimateSuperVertex(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	Timer time("Super Vertex");

	auto& omesh = mesh.getMesh();

	// Setup the vertex list
	auto points = omesh.points();
	size_t vertexCount = omesh.n_vertices();

	Vector3f* d_vertices_ptr;
	size_t verticesSize = vertexCount * sizeof(Vector3f);
	cudaMalloc((void**)&d_vertices_ptr, verticesSize);
	cudaMemcpy(d_vertices_ptr, points, verticesSize, cudaMemcpyHostToDevice);
	
	// Setup the face index
	thrust::host_vector<int> indices(omesh.n_faces() * 3);

	tbb::parallel_for((size_t)0, omesh.n_faces(), [&](const size_t faceId)
	{
		FaceHandle fh(faceId);

		int index = faceId * 3;
		for (auto fv : omesh.fv_range(fh))
		{
			indices[index++] = fv.idx();
		}
	});
	int* d_indices_ptr;
	size_t indicesSize = indices.size() * sizeof(int);
	cudaMalloc((void**)&d_indices_ptr, indicesSize);
	cudaMemcpy(d_indices_ptr, &indices[0], indicesSize, cudaMemcpyHostToDevice);

	// Setup the SV_Header and SV_Data
	thrust::device_vector<SV_Header> d_headers(vertexCount);
	auto d_header_ptr = thrust::raw_pointer_cast(&d_headers[0]);
	thrust::device_vector<SV_Data> d_datas(vertexCount);
	auto d_datas_ptr = thrust::raw_pointer_cast(&d_datas[0]);

	int maxThreadPerBlock = QueryDevice::getMaxThreadPerBlock(0) / 2;
	int blockNeeded = (omesh.n_faces() + (maxThreadPerBlock - 1)) / maxThreadPerBlock;

	std::cout << "Block Needed: " << blockNeeded << std::endl;

	// Setup the header and data
	setupHeaderAndData<<<blockNeeded, maxThreadPerBlock>>>(d_indices_ptr, d_header_ptr, d_datas_ptr, MAX_FACE);
	//CudaCheckError();

	// compute the face quadrics
	thrust::device_vector<Quadric_GPU> FacesQuadric(omesh.n_faces());
	auto d_FaceQuadric_ptr = thrust::raw_pointer_cast(&FacesQuadric[0]);
	computeSVQuadricAllFace<<<blockNeeded, maxThreadPerBlock>>>(d_indices_ptr, d_vertices_ptr, d_FaceQuadric_ptr);
	//CudaCheckError();

	thrust::device_vector<Quadric_GPU> vertexQuadric(vertexCount);
	auto d_vertexQuadric_ptr = thrust::raw_pointer_cast(&vertexQuadric[0]);
	
	blockNeeded = (omesh.n_vertices() + (maxThreadPerBlock - 1)) / maxThreadPerBlock;
	computeSVVertexQuadric<<<blockNeeded, maxThreadPerBlock>>>(d_header_ptr, d_datas_ptr, d_FaceQuadric_ptr, d_vertexQuadric_ptr, vertexCount);
	//CudaCheckError();
	int currentFaceCount = omesh.n_faces();
	//while (targetFaceCount < currentFaceCount)
	{
		thrust::device_vector<bool> d_vertexUsed(vertexCount);
		auto d_vertexUsed_ptr = thrust::raw_pointer_cast(&d_vertexUsed[0]);
		thrust::device_vector<bool> d_independentVertex(vertexCount);
		auto d_independentVertex_ptr = thrust::raw_pointer_cast(&d_independentVertex[0]);

		//blockNeeded = (binSize + (maxThreadPerBlock - 1)) / maxThreadPerBlock;

		selectIndependentVertex<<<blockNeeded, maxThreadPerBlock>>>(d_header_ptr, d_datas_ptr, d_indices_ptr, d_vertexUsed_ptr, d_independentVertex_ptr, vertexCount);
		//CudaCheckError();

		thrust::device_vector<bool> d_checkedvertexUsed(vertexCount);
		auto d_checkedvertexUsed_ptr = thrust::raw_pointer_cast(&d_checkedvertexUsed[0]);
		checkIndependentVertex<<<blockNeeded, maxThreadPerBlock>>>(d_header_ptr, d_datas_ptr, d_indices_ptr, d_independentVertex_ptr, d_checkedvertexUsed_ptr, vertexCount);
		//CudaCheckError();

		thrust::device_vector<BestEdge> d_bestedges(vertexCount);
		auto d_bestEdges_ptr = thrust::raw_pointer_cast(&d_bestedges[0]);
		getBestEdge<<<blockNeeded, maxThreadPerBlock>>>(d_header_ptr, d_datas_ptr, d_indices_ptr, d_vertices_ptr, d_checkedvertexUsed_ptr, d_vertexQuadric_ptr, d_bestEdges_ptr, vertexCount);
		//CudaCheckError();


	}

	time.reportTime();

	return 0;
}