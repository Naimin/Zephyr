#include "OM_Mesh.h"
#include <tbb/parallel_for.h>

using namespace Zephyr;
using namespace Zephyr::Common;

Zephyr::Common::OpenMeshMesh::OpenMeshMesh()
{
}

Zephyr::Common::OpenMeshMesh::OpenMeshMesh(Common::Mesh & mesh)
{
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
		std::vector<OMMesh::VertexHandle> vertexHandles(3);
		for (size_t j = 0; j < 3; ++j)
		{
			vertexHandles[j] = OMMesh::VertexHandle(indices[i]);
		}
		mMesh.add_face(vertexHandles);
	}

	auto edgeCount = mMesh.n_edges();
	auto halfEdgeCount = mMesh.n_halfedges();
}

Zephyr::Common::OpenMeshMesh::OpenMeshMesh(const std::string & path)
{
	OpenMesh::IO::read_mesh(mMesh, path);
}

Zephyr::Common::OpenMeshMesh::~OpenMeshMesh()
{
	
}

OMMesh& Zephyr::Common::OpenMeshMesh::getMesh()
{
	return mMesh;
}
