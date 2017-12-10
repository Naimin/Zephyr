#include "OM_Mesh.h"
#include <tbb/parallel_for.h>

#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

#include <OpenMesh/Core/IO/MeshIO.hh>

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

void Zephyr::Common::OpenMeshMesh::decimate(unsigned targetFaceCount)
{

}
