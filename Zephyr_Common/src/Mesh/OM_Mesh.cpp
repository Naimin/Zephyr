#include "OM_Mesh.h"
#include <tbb/parallel_for.h>

#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalFlippingT.hh>
#include <OpenMesh/Tools/Decimater/ModNormalDeviationT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

#include "../Timer.h"
#include "../Random.h"

using namespace Zephyr;
using namespace Zephyr::Common;

// Decimater type
typedef Decimater::DecimaterT< Zephyr::Common::OMMesh > Decimator;
typedef Decimater::ModQuadricT< Zephyr::Common::OMMesh >::Handle HModQuadric;
typedef Decimater::ModNormalFlippingT< Zephyr::Common::OMMesh >::Handle HModNormalFlipping;
typedef Decimater::ModNormalDeviationT< Zephyr::Common::OMMesh >::Handle HModNormalDeviation;

Zephyr::Common::OpenMeshMesh::OpenMeshMesh()
{
}

Zephyr::Common::OpenMeshMesh::OpenMeshMesh(Common::Mesh & mesh)
{
	loadMesh(mesh);
}

Zephyr::Common::OpenMeshMesh::OpenMeshMesh(const std::string & path)
{
	OpenMesh::IO::read_mesh(mMesh, path);
}

Zephyr::Common::OpenMeshMesh::~OpenMeshMesh()
{
	
}

void Zephyr::Common::OpenMeshMesh::loadMesh(Common::Mesh & mesh)
{
	int previousVerticesCount = (int)mMesh.n_vertices();

	auto vertices = mesh.getVertices();

	// add the vertex from the Graphics::Mesh
	for (size_t i = 0; i < vertices.size(); ++i)
	{
		auto vertex = vertices[i];
		mMesh.add_vertex(OpenMesh::Vec3f(vertex.pos.x(), vertex.pos.y(), vertex.pos.z()));
	}

	auto indices = mesh.getIndices();
	auto faceCount = mesh.getFaceCount();
	for (size_t i = 0; i < faceCount; ++i)
	{
		size_t faceIndex = i * 3;
		std::vector<OMMesh::VertexHandle> vertexHandles(3);
		for (size_t j = 0; j < 3; ++j)
		{
			int index = indices[faceIndex + j] + previousVerticesCount;
			vertexHandles[j] = OMMesh::VertexHandle(index);
		}

		// if a invalid face (ie, line or point) skip
		if (vertexHandles[0] == vertexHandles[1] || vertexHandles[0] == vertexHandles[2] || vertexHandles[1] == vertexHandles[2])
			continue;

		mMesh.add_face(vertexHandles[0], vertexHandles[1], vertexHandles[2]);
	}
}

OMMesh& Zephyr::Common::OpenMeshMesh::getMesh()
{
	return mMesh;
}

bool Common::OpenMeshMesh::exports(const std::string & path)
{
	if (!OpenMesh::IO::write_mesh(mMesh, path))
	{
		std::cerr << "write error\n";
		return false;
	}

	std::cout << "Export " << path << " done." << std::endl;
	return true;
}

int Zephyr::Common::OpenMeshMesh::decimate(unsigned int targetFaceCount, DecimationType type)
{
	Timer timer;
	int collapseCount = -1;
	auto previousFaceCount = mMesh.n_faces();
	std::cout << "Using ";
	if (RANDOM_DECIMATE == type)
	{
		int binCount = 8;
		std::cout << "Random Decimation..." << std::endl;
		collapseCount = decimateRandom(targetFaceCount, binCount);
	}
	else
	{
		std::cout << "Greedy Decimation..." << std::endl;
		collapseCount = decimateGreedy(targetFaceCount);
	}
	auto elapseTime = timer.getElapsedTime();

	std::cout << "Decimation done in " << elapseTime << " sec" << std::endl;
	std::cout << "Original Face Count: " << previousFaceCount << std::endl;
	std::cout << "Target Face Count: " << targetFaceCount << std::endl;
	std::cout << "Removed Face Count: " << collapseCount << std::endl;
	std::cout << "Decimated Face Count: " << mMesh.n_faces() << std::endl;
	std::cout << "Percentage decimated: " << ((previousFaceCount - mMesh.n_faces()) / (float)previousFaceCount) * 100.0f << " %" << std::endl << std::endl;

	exports("D:\\sandbox\\decimatedMesh.obj");

	return collapseCount; 
}

int Zephyr::Common::OpenMeshMesh::decimateGreedy(unsigned int targetFaceCount)
{
	const float maxQuadricError = 0.001f;
	const float maxNormalFlipDeviation = 45.0f;
	const float maxNormalDeviation = 15.0f;

	Decimator decimator(mMesh);
	
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

	size_t numOfCollapseRequired = mMesh.n_faces() - targetFaceCount;

	size_t actualNumOfCollapse = decimator.decimate(numOfCollapseRequired);

	mMesh.garbage_collection();

	return (int)actualNumOfCollapse;
}

int Zephyr::Common::OpenMeshMesh::decimateRandom(unsigned int targetFaceCount, int binSize)
{
	int retryCount = 0;
	size_t previousFaceCount = mMesh.n_faces();
	size_t totalHalfEdgeCount = mMesh.n_halfedges();

	int numOfThreads = std::thread::hardware_concurrency();
	std::vector<std::vector<HalfedgeHandle>> selectedEdgesPerThread(numOfThreads);
	std::vector<std::shared_ptr<RandomGenerator>> randomGenerators;

	// setup the random generator and selectedEdge memory
	for (int threadId = 0; threadId < numOfThreads; ++threadId)
	{
		randomGenerators.push_back(std::shared_ptr<RandomGenerator>(new RandomGenerator(0, (int)totalHalfEdgeCount-1, threadId)));
		selectedEdgesPerThread[threadId].resize(binSize);
	}

	int totalCollapseCount = 0;
	while (retryCount < 10 && mMesh.n_faces() > targetFaceCount)
	{
		// parallelly select edge to collapse
		tbb::parallel_for(0, numOfThreads, [&](const int threadId)
		{
			auto& selectedEdges = selectedEdgesPerThread[threadId];

			for (int selection = 0; selection < binSize; ++selection)
			{
				HalfedgeHandle halfEdgeHandle(randomGenerators[threadId]->next());

				if (mMesh.status(halfEdgeHandle).deleted() || !mMesh.is_collapse_ok(halfEdgeHandle))
				{
					selectedEdges[selection] = HalfedgeHandle(-1);
					continue;
				}

				// compute the quadric error here

				selectedEdges[selection] = halfEdgeHandle;
			}
		});
		
		// perform the collapse
		int collapseCount = 0;
		for (int threadId = 0; threadId < numOfThreads; ++threadId)
		{
			auto& selectedEdges = selectedEdgesPerThread[threadId];
			for (auto halfEdgeHandle : selectedEdges)
			{
				if (!halfEdgeHandle.is_valid())
					continue;

				if (mMesh.status(halfEdgeHandle).deleted() || !mMesh.is_collapse_ok(halfEdgeHandle))
				{
					continue;
				}

				auto halfEdge = mMesh.halfedge(halfEdgeHandle);
				mMesh.collapse(halfEdgeHandle);
				++collapseCount;
			}
		}
		totalCollapseCount += collapseCount;

		//std::cout << "Collapsed this iteration: " << collapseCount << std::endl;

		// if there is no changes in face count, retry
		if(0 == collapseCount)
		{ 
			++retryCount;
		//	std::cout << "Retrying: " << retryCount << std::endl;
		}

		previousFaceCount = mMesh.n_faces() - totalCollapseCount;
	}
	
	mMesh.garbage_collection();
	return 0;
}