#ifndef ZEPHYR_GPU_OpenMesh_H
#define ZEPHYR_GPU_OpenMesh_H

#include "../stdfx.h"
#include <Mesh/OM_Mesh.h>
#include <thrust/host_vector.h>

namespace Zephyr
{
	namespace GPU
	{
		typedef unsigned char INDEX_TYPE;

		struct SortVector3f : public Common::Vector3f
		{
			SortVector3f(float x=0, float y=0, float z=0) : Common::Vector3f(x, y, z) {}

			bool operator<(const SortVector3f& b) const
			{
				return std::tie(x(), y(), z()) < std::tie(b.x(), b.y(), b.z());
			}
		};

		const INDEX_TYPE MAX_VALENCE = 32;
		const INDEX_TYPE MAX_FACE = 48;

		struct QEM_Data
		{
			QEM_Data() : bValid(true), vertexCount(0), indexCount(0), vertexToKeepId(-1) {}

			Common::Vector3f vertices[MAX_VALENCE]; // vertex buffer
			INDEX_TYPE indices[MAX_FACE*3]; // all the faces formed
			INDEX_TYPE vertexCount;
			INDEX_TYPE indexCount;
			INDEX_TYPE vertexToKeepId;
			bool bValid;

			void reset()
			{
				bValid = true;
				vertexCount = 0;
				indexCount = 0;
				vertexToKeepId = -1;
			}
		};

		struct OpenMesh_GPU
		{
			OpenMesh_GPU(size_t totalSelectionCount);
			std::vector<QEM_Data>* copyPartialMesh(Common::OMMesh& mesh, const thrust::host_vector<int>& randomList);
		
			std::vector<QEM_Data>* getQEM_Data();

		protected:
			void collectOneRingNeighbour(VertexHandle vh, Common::OMMesh & mesh, Common::Vector3f* vertexBuffer, INDEX_TYPE& vertexCount, INDEX_TYPE* indexBuffer, INDEX_TYPE& indexCount, std::map<SortVector3f, INDEX_TYPE>& uniqueVertex);

			std::vector<QEM_Data> mQEM_Data;
		};
	}
}

#endif