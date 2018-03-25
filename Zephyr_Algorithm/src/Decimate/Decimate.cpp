#include "Decimate.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/mutex.h>
#include <map>

#include <Timer.h>
#include <Random.h>

#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalDeviationT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

#include "QuadricError.h"

//#define nearest_collapse 0

using namespace Zephyr;
using namespace Zephyr::Common;

// Decimater type
typedef Decimater::DecimaterT< Zephyr::Common::OMMesh > Decimator;
typedef Decimater::ModQuadricT< Zephyr::Common::OMMesh >::Handle HModQuadric;
typedef Decimater::ModNormalFlippingT< Zephyr::Common::OMMesh >::Handle HModNormalFlipping;
typedef Decimater::ModNormalDeviationT< Zephyr::Common::OMMesh >::Handle HModNormalDeviation;

int Zephyr::Algorithm::Decimater::decimate(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, int binSize, DecimationType type)
{
	auto& omesh = mesh.getMesh();

	Timer timer;
	int collapseCount = -1;
	auto previousFaceCount = omesh.n_faces();

	std::cout << "Using ";
	if (RANDOM_DECIMATE == type)
	{
		std::cout << "Random Decimation..." << std::endl;
		collapseCount = decimateRandom(mesh, targetFaceCount, binSize);
	}
	else if (RANDOM_DECIMATE_VERTEX == type)
	{
		std::cout << "Random Decimation Vertex..." << std::endl;
		collapseCount = decimateRandomVertex(mesh, targetFaceCount, binSize);
	}
	else if(GREEDY_DECIMATE == type)
	{
		std::cout << "Greedy Decimation..." << std::endl;
		collapseCount = decimateGreedy(mesh, targetFaceCount);
	}
	else if (ADAPTIVE_RANDOM_DECIMATE == type)
	{
		std::cout << "Adaptive Random Decimation..." << std::endl;
		collapseCount = decimateAdaptiveRandom(mesh, targetFaceCount, binSize);
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

int Zephyr::Algorithm::Decimater::decimateGreedy(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount)
{
	const double maxQuadricError = 0.1;
	const double maxNormalFlipDeviation = 45.0;
	const double maxNormalDeviation = 15.0;

	auto& omesh = mesh.getMesh();

	Decimator decimator(omesh);

	HModQuadric hModQuadric;      // use a quadric module
	decimator.add(hModQuadric); // register module at the decimator
	decimator.module(hModQuadric).set_max_err(maxQuadricError);

	HModNormalFlipping hModNormalFlipping; // use a module that prevent flipping of triangle
	decimator.add(hModNormalFlipping); 					// register the normal flipping module
	decimator.module(hModNormalFlipping).set_max_normal_deviation(maxNormalFlipDeviation);	// set the maximum normal deviation 

	HModNormalDeviation hModNormalDeviation; // use a module that prevent large change in face normal
	decimator.add(hModNormalDeviation); // register module to the decimator
	decimator.module(hModNormalDeviation).set_binary(true); // set to true if not the main decimation score
	decimator.module(hModNormalDeviation).set_normal_deviation((float)maxNormalDeviation);

	if (!decimator.initialize())
		return -1;

	size_t numOfCollapseRequired = (omesh.n_faces() - targetFaceCount) / 2;

	size_t actualNumOfCollapse = decimator.decimate(numOfCollapseRequired);

	omesh.garbage_collection();

	return (int)actualNumOfCollapse;
}

int Zephyr::Algorithm::Decimater::decimateRandom(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	const float maxQuadricError = 0.1f;
	const float maxNormalFlipDeviation = 45.0;
	const int maxRetryCount = 250;

	auto& omesh = mesh.getMesh();

	int retryCount = 0;
	size_t initialFaceCount = omesh.n_faces();
	size_t currentFaceCount = initialFaceCount;
	size_t totalHalfEdgeCount = omesh.n_halfedges();

	int numOfThreads = std::thread::hardware_concurrency();
	std::vector<std::vector<std::pair<float, HalfedgeHandle>>> selectedErrorEdgesPerThread(numOfThreads);
	std::vector<std::shared_ptr<RandomGenerator>> randomGenerators;

	// setup the random generator and selectedEdge memory
	for (int threadId = 0; threadId < numOfThreads; ++threadId)
	{
		randomGenerators.push_back(std::shared_ptr<RandomGenerator>(new RandomGenerator(0, (int)totalHalfEdgeCount - 1, threadId)));
		selectedErrorEdgesPerThread[threadId].resize(binSize);
	}

	int totalCollapseCount = 0;
	while (retryCount < maxRetryCount && currentFaceCount > targetFaceCount)
	{
		// parallelly select edge to collapse
		tbb::parallel_for(0, numOfThreads, [&](const int threadId)
		{
			auto& selectedEdges = selectedErrorEdgesPerThread[threadId];

			for (int selection = 0; selection < (int)binSize; ++selection)
			{
				HalfedgeHandle halfEdgeHandle(randomGenerators[threadId]->next());

				if (omesh.status(halfEdgeHandle).deleted() || !omesh.is_collapse_ok(halfEdgeHandle))
				{
					// set the invalid value
					selectedEdges[selection] = std::make_pair(Algorithm::INVALID_COLLAPSE, halfEdgeHandle);
					continue;
				}

				// compute the quadric error here
				float quadricError = QuadricError::computeQuadricError(halfEdgeHandle, mesh, maxQuadricError, maxNormalFlipDeviation);

				selectedEdges[selection] = std::make_pair(quadricError, halfEdgeHandle);
			}
		});

		// Get the lowest error edge
		tbb::mutex mutex;
		std::vector<int> bestEdges(numOfThreads, -1);
		std::vector<float> bestEdgesError(numOfThreads, -1);
		tbb::parallel_for(0, numOfThreads, [&](const int threadId)
		{
			auto& selectedEdges = selectedErrorEdgesPerThread[threadId];
			
			HalfedgeHandle bestEdge;
			float bestError = Algorithm::INVALID_COLLAPSE;
			for (auto errorEdge : selectedEdges)
			{
				auto error = errorEdge.first;
				auto halfEdgeHandle = errorEdge.second;

				if (errorEdge.first == Algorithm::INVALID_COLLAPSE || !errorEdge.second.is_valid())
					continue;

				if (omesh.status(halfEdgeHandle).deleted() || !omesh.is_collapse_ok(halfEdgeHandle))
					continue;

				if (error < bestError)
				{
					bestError = error;
					bestEdge = halfEdgeHandle;
				}
			}

			if (bestEdge.is_valid())
			{
				bestEdges[threadId] = bestEdge.idx();
				bestEdgesError[threadId] = bestError;
			}
		});

		// do the exact collapse
		int collapseCount = 0;
		int originalCollapse = 0;
		int additionalCollapse = 0;

		int faceCollapsed = 0;
		int threadCount = -1;
		for (auto bestEdgeId : bestEdges)
		{
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

#ifdef nearest_collapse
			// Perform another collapse in the vicinity of the previous collapse
			auto toVertex = omesh.to_vertex_handle(halfEdgeHandle);
			
			// Only allow the QEM error to be within 10% more than the original edge collapse's QEM
			// This prevent collapsing of edges that further aggrevate the density mismatch problem.
			tbb::atomic<float> minError = bestEdgesError[threadCount] * 1.1f;
			tbb::mutex mutex;
			HalfedgeHandle lowestErrorEdge;

			//for (OMMesh::VertexVertexIter vv_it = omesh.vv_begin(toVertex); vv_it.is_valid(); ++vv_it)
			//tbb::parallel_for_each(omesh.vv_begin(toVertex), omesh.vv_end(toVertex), [&](VertexHandle vh)
			{
				// incoming half edge to the newly collapsed vertex
				for (OMMesh::VertexIHalfedgeIter ve_it = omesh.vih_begin(toVertex); ve_it.is_valid(); ++ve_it)
				{
					float QEM = QuadricError::computeQuadricError(*ve_it, mesh, maxQuadricError, maxNormalFlipDeviation);
					if (minError > QEM)
					{
						minError = QEM;
						tbb::mutex::scoped_lock lock(mutex);
						lowestErrorEdge = *ve_it;
					}
				}
				// outgoing half edge to the newly collapsed vertex
				for (OMMesh::VertexOHalfedgeIter ve_it = omesh.voh_begin(toVertex); ve_it.is_valid(); ++ve_it)
				{
					float QEM = QuadricError::computeQuadricError(*ve_it, mesh, maxQuadricError, maxNormalFlipDeviation);
					if (minError > QEM)
					{
						minError = QEM;
						tbb::mutex::scoped_lock lock(mutex);
						lowestErrorEdge = *ve_it;
					}
				}
			}//);

			if (!lowestErrorEdge.is_valid() || !omesh.is_collapse_ok(lowestErrorEdge))
				continue;

			omesh.collapse(lowestErrorEdge);
			++collapseCount;
			++additionalCollapse;
#endif

		}
		totalCollapseCount += collapseCount;

		//std::cout << "Total Collapsed this iteration: " << collapseCount << " Total Collapsed: " << totalCollapseCount << std::endl;
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

int Zephyr::Algorithm::Decimater::decimateRandomVertex(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	const float maxQuadricError = 0.1f;
	const float maxNormalFlipDeviation = 45.0f;
	const int maxRetryCount = 500;

	auto& omesh = mesh.getMesh();

	int retryCount = 0;
	size_t initialFaceCount = omesh.n_faces();
	size_t currentFaceCount = initialFaceCount;
	size_t totalVertexCount = omesh.n_vertices();

	int numOfThreads = std::thread::hardware_concurrency();
	std::vector<std::vector<std::pair<float, HalfedgeHandle>>> selectedErrorEdgesPerThread(numOfThreads);
	std::vector<std::shared_ptr<RandomGenerator>> randomGenerators;

	// setup the random generator and selectedEdge memory
	for (int threadId = 0; threadId < numOfThreads; ++threadId)
	{
		randomGenerators.push_back(std::shared_ptr<RandomGenerator>(new RandomGenerator(0, (int)totalVertexCount - 1, threadId)));
		selectedErrorEdgesPerThread[threadId].resize(binSize);
	}

	int totalCollapseCount = 0;
	while (retryCount < maxRetryCount && currentFaceCount > targetFaceCount)
	{
		// parallelly select edge to collapse
		tbb::parallel_for(0, numOfThreads, [&](const int threadId)
		{
			auto& selectedEdges = selectedErrorEdgesPerThread[threadId];

			for (int selection = 0; selection < (int)binSize; ++selection)
			{
				VertexHandle vertexHandle(randomGenerators[threadId]->next());

				if (omesh.status(vertexHandle).deleted())
				{
					continue;
				}

				// Only allow the QEM error to be within 10% more than the original edge collapse's QEM
				// This prevent collapsing of edges that further aggrevate the density mismatch problem.
				tbb::atomic<float> minError = std::numeric_limits<float>::max();
				tbb::mutex mutex;
				HalfedgeHandle lowestErrorEdge;

				//for (OMMesh::VertexVertexIter vv_it = omesh.vv_begin(toVertex); vv_it.is_valid(); ++vv_it)
				{
					// incoming half edge to the newly collapsed vertex
					for (OMMesh::VertexIHalfedgeIter ve_it = omesh.vih_begin(vertexHandle); ve_it.is_valid(); ++ve_it)
					{
						float QEM = QuadricError::computeQuadricError(*ve_it, mesh, maxQuadricError, maxNormalFlipDeviation);
						if (minError > QEM)
						{
							minError = QEM;
							tbb::mutex::scoped_lock lock(mutex);
							lowestErrorEdge = *ve_it;
						}
					}
					// outgoing half edge to the newly collapsed vertex
					for (OMMesh::VertexOHalfedgeIter ve_it = omesh.voh_begin(vertexHandle); ve_it.is_valid(); ++ve_it)
					{
						float QEM = QuadricError::computeQuadricError(*ve_it, mesh, maxQuadricError, maxNormalFlipDeviation);
						if (minError > QEM)
						{
							minError = QEM;
							tbb::mutex::scoped_lock lock(mutex);
							lowestErrorEdge = *ve_it;
						}
					}
				}//);

				if (omesh.status(vertexHandle).deleted() || !omesh.is_collapse_ok(lowestErrorEdge))
				{
					// set the invalid value
					selectedEdges[selection] = std::make_pair(Algorithm::INVALID_COLLAPSE, lowestErrorEdge);
					continue;
				}

				// compute the quadric error here
				float quadricError = QuadricError::computeQuadricError(lowestErrorEdge, mesh, maxQuadricError, maxNormalFlipDeviation);

				selectedEdges[selection] = std::make_pair(quadricError, lowestErrorEdge);
			}
		});

		// Get the lowest error edge
		tbb::mutex mutex;
		std::vector<int> bestEdges(numOfThreads, -1);
		std::vector<float> bestEdgesError(numOfThreads, -1);
		tbb::parallel_for(0, numOfThreads, [&](const int threadId)
		{
			auto& selectedEdges = selectedErrorEdgesPerThread[threadId];

			HalfedgeHandle bestEdge;
			float bestError = Algorithm::INVALID_COLLAPSE;
			for (auto errorEdge : selectedEdges)
			{
				auto error = errorEdge.first;
				auto halfEdgeHandle = errorEdge.second;

				if (errorEdge.first == Algorithm::INVALID_COLLAPSE || !errorEdge.second.is_valid())
					continue;

				if (omesh.status(halfEdgeHandle).deleted() || !omesh.is_collapse_ok(halfEdgeHandle))
					continue;

				if (error < bestError)
				{
					bestError = error;
					bestEdge = halfEdgeHandle;
				}
			}

			if (bestEdge.is_valid())
			{
				bestEdges[threadId] = bestEdge.idx();
				bestEdgesError[threadId] = bestError;
			}
		});

		// do the exact collapse
		int collapseCount = 0;
		int originalCollapse = 0;
		int additionalCollapse = 0;

		int faceCollapsed = 0;
		int threadCount = -1;
		for (auto bestEdgeId : bestEdges)
		{
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
/*
#ifdef nearest_collapse
			// Perform another collapse in the vicinity of the previous collapse
			auto toVertex = omesh.to_vertex_handle(halfEdgeHandle);

			// Only allow the QEM error to be within 10% more than the original edge collapse's QEM
			// This prevent collapsing of edges that further aggrevate the density mismatch problem.
			tbb::atomic<float> minError = bestEdgesError[threadCount] * 1.1f;
			tbb::mutex mutex;
			HalfedgeHandle lowestErrorEdge;

			//for (OMMesh::VertexVertexIter vv_it = omesh.vv_begin(toVertex); vv_it.is_valid(); ++vv_it)
			//tbb::parallel_for_each(omesh.vv_begin(toVertex), omesh.vv_end(toVertex), [&](VertexHandle vh)
			{
				// incoming half edge to the newly collapsed vertex
				for (OMMesh::VertexIHalfedgeIter ve_it = omesh.vih_begin(toVertex); ve_it.is_valid(); ++ve_it)
				{
					float QEM = QuadricError::computeQuadricError(*ve_it, mesh, maxQuadricError, maxNormalFlipDeviation);
					if (minError > QEM)
					{
						minError = QEM;
						tbb::mutex::scoped_lock lock(mutex);
						lowestErrorEdge = *ve_it;
					}
				}
				// outgoing half edge to the newly collapsed vertex
				for (OMMesh::VertexOHalfedgeIter ve_it = omesh.voh_begin(toVertex); ve_it.is_valid(); ++ve_it)
				{
					float QEM = QuadricError::computeQuadricError(*ve_it, mesh, maxQuadricError, maxNormalFlipDeviation);
					if (minError > QEM)
					{
						minError = QEM;
						tbb::mutex::scoped_lock lock(mutex);
						lowestErrorEdge = *ve_it;
					}
				}
			}//);

			if (!lowestErrorEdge.is_valid() || !omesh.is_collapse_ok(lowestErrorEdge))
				continue;

			omesh.collapse(lowestErrorEdge);
			++collapseCount;
			++additionalCollapse;
#endif
			*/
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


int Zephyr::Algorithm::Decimater::decimateAdaptiveRandom(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	const int maxRetryCount = 10;

	auto& omesh = mesh.getMesh();

	int totalCollapseCount = 0;

	int retryCount = 0;
	while (retryCount < maxRetryCount && omesh.n_faces() > targetFaceCount)
	{
		int adaptiveBinSize = binSize + retryCount; // for each incomplete run of the decimateRandom, we increase the bin size by 1
		totalCollapseCount += decimateRandom(mesh, targetFaceCount, adaptiveBinSize);

		++retryCount;
	}
	
	return totalCollapseCount;
}
