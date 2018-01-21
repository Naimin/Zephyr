#include "Decimate.h"

#include <tbb/parallel_for.h>
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

using namespace Zephyr;
using namespace Zephyr::Common;

// Decimater type
typedef Decimater::DecimaterT< Zephyr::Common::OMMesh > Decimator;
typedef Decimater::ModQuadricT< Zephyr::Common::OMMesh >::Handle HModQuadric;
typedef Decimater::ModNormalFlippingT< Zephyr::Common::OMMesh >::Handle HModNormalFlipping;
typedef Decimater::ModNormalDeviationT< Zephyr::Common::OMMesh >::Handle HModNormalDeviation;

int Zephyr::Algorithm::Decimater::decimate(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, DecimationType type)
{
	auto& omesh = mesh.getMesh();

	Timer timer;
	int collapseCount = -1;
	auto previousFaceCount = omesh.n_faces();
	std::cout << "Using ";
	if (RANDOM_DECIMATE == type)
	{
		int binCount = 8;
		std::cout << "Random Decimation..." << std::endl;
		collapseCount = decimateRandom(mesh, targetFaceCount, binCount);
	}
	else
	{
		std::cout << "Greedy Decimation..." << std::endl;
		collapseCount = decimateGreedy(mesh, targetFaceCount);
	}
	auto elapseTime = timer.getElapsedTime();

	auto& omeshDecimated = mesh.getMesh();

	std::cout << "Decimation done in " << elapseTime << " sec" << std::endl;
	std::cout << "Original Face Count: " << previousFaceCount << std::endl;
	std::cout << "Target Face Count: " << targetFaceCount << std::endl;
	std::cout << "Removed Face Count: " << collapseCount << std::endl;
	std::cout << "Decimated Face Count: " << omeshDecimated.n_faces() << std::endl;
	std::cout << "Percentage decimated: " << ((previousFaceCount - omeshDecimated.n_faces()) / (float)previousFaceCount) * 100.0f << " %" << std::endl << std::endl;

	return collapseCount;
}

int Zephyr::Algorithm::Decimater::decimateGreedy(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount)
{
	const float maxQuadricError = 0.01f;
	const float maxNormalFlipDeviation = 45.0f;
	const float maxNormalDeviation = 15.0f;

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
	decimator.module(hModNormalDeviation).set_normal_deviation(maxNormalDeviation);

	if (!decimator.initialize())
		return -1;

	size_t numOfCollapseRequired = omesh.n_faces() - targetFaceCount;

	size_t actualNumOfCollapse = decimator.decimate(numOfCollapseRequired);

	omesh.garbage_collection();

	return (int)actualNumOfCollapse;
}

int Zephyr::Algorithm::Decimater::decimateRandom(Common::OpenMeshMesh & mesh, unsigned int targetFaceCount, unsigned int binSize)
{
	const float maxQuadricError = 0.01f;
	const float maxNormalFlipDeviation = 15.0f;

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
	while (retryCount < 500 && currentFaceCount > targetFaceCount)
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
		std::vector<int> bestEdgesTemp(numOfThreads, -1);
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
				bestEdgesTemp[threadId] = bestEdge.idx();
			}
		});

		std::set<int> bestEdges;
		bestEdges.insert(bestEdgesTemp.begin(), bestEdgesTemp.end());
		bestEdges.erase(-1);

		// do the exact collapse
		int collapseCount = 0;
		for (auto bestEdgeId : bestEdges)
		{
			HalfedgeHandle halfEdgeHandle(bestEdgeId);
			if (!omesh.is_collapse_ok(halfEdgeHandle))
				continue;

			omesh.collapse(halfEdgeHandle);
			++collapseCount;
		}
		totalCollapseCount += collapseCount;

		//std::cout << "Collapsed this iteration: " << collapseCount << std::endl;
		//std::cout << "Total Collapsed : " << totalCollapseCount << std::endl;

		// if there is no changes in face count, retry
		if (0 == collapseCount)
		{
			++retryCount;
			//std::cout << "Retrying: " << retryCount << std::endl;
		}

		currentFaceCount -= collapseCount * 2; // each collapse remove 2 face
	}
	omesh.garbage_collection();

	return totalCollapseCount;
}
