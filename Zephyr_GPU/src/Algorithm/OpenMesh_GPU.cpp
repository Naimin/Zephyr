#include "OpenMesh_GPU.h"
#include <Decimate/QuadricError.h>
#include <tbb/parallel_for.h>
#include <map>

using namespace Zephyr;
using namespace Zephyr::GPU;
using namespace Zephyr::Common;

void OpenMesh_GPU::collectOneRingNeighbour(VertexHandle vh, Common::OMMesh& mesh, std::vector<Common::Vector3f>& vertexBuffer, std::vector<INDEX_TYPE>& indexBuffer, std::map<SortVector3f, INDEX_TYPE>& uniqueVertex)
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
				vertexBuffer.push_back(vertices[i]);
				uniqueVertex.insert(std::make_pair(vertices[i], uniqueVertex.size()));
				uniqueItr = uniqueVertex.find(vertices[i]);
			}
			indexBuffer.push_back(uniqueItr->second);
		}
	}
}

std::vector<QEM_Data> OpenMesh_GPU::copyPartialMesh(Common::OMMesh& mesh, const std::vector<int>& randomList)
{
	// pre-allocate
	std::vector<QEM_Data> QEM_Datas(randomList.size());

	// collect only the information needed to compute all the quadric
	tbb::parallel_for(0, (int)randomList.size(), [&](const int idx)
	{
		int randomNum = randomList[idx];
		CollapseInfo collapseInfo(mesh, HalfedgeHandle(randomNum));

		QEM_Data& QEM_Data = QEM_Datas[idx];

		std::vector<Common::Vector3f> vertexBuffer;
		std::vector<INDEX_TYPE> indexBuffer;

		std::map<SortVector3f, INDEX_TYPE> uniqueVertex;

		// Collect all the vertices
		collectOneRingNeighbour(collapseInfo.v0, mesh, vertexBuffer, indexBuffer, uniqueVertex);
		collectOneRingNeighbour(collapseInfo.v1, mesh, vertexBuffer, indexBuffer, uniqueVertex);

		QEM_Data.indices.swap(indexBuffer);
		QEM_Data.vertices.swap(vertexBuffer);
	});

	return QEM_Datas;
}