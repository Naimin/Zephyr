#include "OM_Mesh.h"
#include <tbb/parallel_for.h>

using namespace Zephyr;
using namespace Zephyr::Algorithm;

Zephyr::Algorithm::OpenMeshMesh::OpenMeshMesh()
{
}

Zephyr::Algorithm::OpenMeshMesh::OpenMeshMesh(Graphics::Mesh & mesh)
{
	auto vertices = mesh.getVertices();
	
	// add the vertex from the Graphics::Mesh
	for (size_t i = 0; i < vertices.size(); ++i)
	{
		auto vertex = vertices[i];
		mMesh.add_vertex(OpenMesh::Vec3f(vertex.pos.x, vertex.pos.y, vertex.pos.z));
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
}

Zephyr::Algorithm::OpenMeshMesh::~OpenMeshMesh()
{
	
}
