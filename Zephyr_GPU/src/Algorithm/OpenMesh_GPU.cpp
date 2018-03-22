#include "OpenMesh_GPU.h"
#include <Decimate/QuadricError.h>
#include <tbb/parallel_for.h>
#include <map>
#include <Timer.h>

using namespace Zephyr;
using namespace Zephyr::GPU;
using namespace Zephyr::Common;

void OpenMesh_GPU::collectOneRingNeighbour(VertexHandle vh, 
										   Common::OMMesh& mesh, 
										   Common::Vector3f* vertexBuffer, 
										   INDEX_TYPE& vertexCount,
										   INDEX_TYPE* indexBuffer, 
										   INDEX_TYPE& indexCount,
										   std::map<SortVector3f, INDEX_TYPE>& uniqueVertex)
{
	// For each face of the vertex iterate over it to collect them
	for (OMMesh::VertexFaceIter vf_Itr = mesh.vf_iter(vh); vf_Itr.is_valid(); ++vf_Itr)
	{
		auto fv_it = mesh.fv_iter(*vf_Itr);
		VertexHandle vh[3];
		vh[0] = *fv_it; ++fv_it;
		vh[1] = *fv_it; ++fv_it;
		vh[2] = *fv_it;
		
		SortVector3f vertices[3];
		for (int i = 0; i < 3; ++i)
		{
			auto point = mesh.point(vh[i]);
			vertices[i] = SortVector3f(point[0], point[1], point[2]);

			// if a new vertex
			auto uniqueItr = uniqueVertex.find(vertices[i]);
			if (uniqueVertex.end() == uniqueItr)
			{
				vertexBuffer[vertexCount++] = vertices[i];
				uniqueVertex.insert(std::make_pair(vertices[i], uniqueVertex.size()));
				uniqueItr = uniqueVertex.find(vertices[i]);
			}
			indexBuffer[indexCount++] = uniqueItr->second;
		}
	}
}

Zephyr::GPU::OpenMesh_GPU::OpenMesh_GPU(size_t totalSelectionCount) : mQEM_Data(totalSelectionCount)
{
}

std::vector<QEM_Data>* OpenMesh_GPU::copyPartialMesh(Common::OMMesh& mesh, const thrust::host_vector<int>& randomList)
{
	// collect only the information needed to compute all the quadric
	tbb::parallel_for(0, (int)randomList.size(), [&](const int idx)
	{
		int randomNum = randomList[idx];
		QEM_Data& data = mQEM_Data[idx];

		// reset and reuse
		data.reset();

		HalfedgeHandle halfEdgeHandle(randomNum);
		if (mesh.status(halfEdgeHandle).deleted() || !mesh.is_collapse_ok(halfEdgeHandle))
		{
			data.bValid = false;
			return;
		}

		CollapseInfo collapseInfo(mesh, halfEdgeHandle);
		float angleError = Algorithm::QuadricError::computeTriangleFlipAngle(collapseInfo, mesh, 15.0f);
		if (angleError == Algorithm::INVALID_COLLAPSE)
		{
			data.bValid = false;
			return;
		}

		std::map<SortVector3f, INDEX_TYPE> uniqueVertex;

		// Collect all the vertices
		// TODO: need proper bound check if vertex go beyond MAX_VALENCE and similarly MAX_FACE
		collectOneRingNeighbour(collapseInfo.v0, mesh, data.vertices, data.vertexCount, data.indices, data.indexCount, uniqueVertex);
		collectOneRingNeighbour(collapseInfo.v1, mesh, data.vertices, data.vertexCount, data.indices, data.indexCount, uniqueVertex);
		
		auto pointToKeep = mesh.point(collapseInfo.v1);
		data.vertexToKeepId = uniqueVertex[SortVector3f(pointToKeep[0], pointToKeep[1], pointToKeep[2])];
	});
	return &mQEM_Data;
}

std::vector<QEM_Data>* Zephyr::GPU::OpenMesh_GPU::getQEM_Data()
{
	return &mQEM_Data;
}
