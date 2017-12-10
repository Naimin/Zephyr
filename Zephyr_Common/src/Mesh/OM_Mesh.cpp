#include "OM_Mesh.h"
#include <tbb/parallel_for.h>

#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

#include <OpenMesh/Core/IO/MeshIO.hh>

#include "../Timer.h"

using namespace Zephyr;
using namespace Zephyr::Common;

// Decimater type
typedef Decimater::DecimaterT< Zephyr::Common::OMMesh > Decimator;
typedef Decimater::ModQuadricT< Zephyr::Common::OMMesh >::Handle HModQuadric;

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

	auto edgeCount = mMesh.n_edges();
	auto halfEdgeCount = mMesh.n_halfedges();

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
		std::cout << "Random Decimation...";
		collapseCount = decimateRandom(targetFaceCount);
	}
	else
	{
		std::cout << "Greedy Decimation...";
		collapseCount = decimateGreedy(targetFaceCount);
	}
	mMesh.garbage_collection();
	auto elapseTime = timer.getElapsedTime();

	std::cout << "Done in " << elapseTime << " sec" << std::endl;
	std::cout << "Original Face Count: " << previousFaceCount << std::endl;
	std::cout << "Target Face Count: " << targetFaceCount << std::endl;
	std::cout << "Decimated Face Count: " << mMesh.n_faces() << std::endl;
	std::cout << "Percentage decimated: " << mMesh.n_faces() / (float)previousFaceCount << " %" << std::endl;

	exports("D:\\sandbox\\decimatedMesh.obj");

	return collapseCount;
}

int Zephyr::Common::OpenMeshMesh::decimateGreedy(unsigned int targetFaceCount)
{
	Decimator decimator(mMesh);
	HModQuadric hModQuadric;      // use a quadric module

	decimator.add(hModQuadric); // register module at the decimater
	
	/*
	* since we need exactly one priority module (non-binary)
	* we have to call set_binary(false) for our priority module
	* in the case of HModQuadric, unset_max_err() calls set_binary(false) internally
	*/
	//
	//decimator.module(hModQuadric).unset_max_err();
	decimator.initialize();

	size_t numOfCollapseRequired = mMesh.n_faces() - targetFaceCount;

	size_t actualNumOfCollapse = decimator.decimate(numOfCollapseRequired);

	return (int)actualNumOfCollapse;
}

int Zephyr::Common::OpenMeshMesh::decimateRandom(unsigned int targetFaceCount)
{
	return 0;
}