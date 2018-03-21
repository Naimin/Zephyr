#ifndef ZEPHYR_GPU_OpenMesh_H
#define ZEPHYR_GPU_OpenMesh_H

#include "../stdfx.h"
#include <Mesh/OM_Mesh.h>

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

		const INDEX_TYPE MAX_VALENCE = 16;
		const INDEX_TYPE MAX_FACE = 32;

		struct QEM_Data
		{
			QEM_Data() : bValid(true) {}

			Common::Vector3f vertices[MAX_VALENCE]; // vertex buffer
			INDEX_TYPE indices[MAX_FACE*3]; // all the faces formed
			INDEX_TYPE vertexCount;
			INDEX_TYPE indexCount;
			INDEX_TYPE vertexToKeepId;
			bool bValid;
		};

		struct OpenMesh_GPU
		{
			static std::vector<QEM_Data> copyPartialMesh(Common::OMMesh& mesh, const std::vector<int>& randomList);
		
		protected:
			static void collectOneRingNeighbour(VertexHandle vh, Common::OMMesh & mesh, std::vector<Common::Vector3f>& vertexBuffer, std::vector<INDEX_TYPE>& indexBuffer, std::map<SortVector3f, INDEX_TYPE>& uniqueVertex);
		};
	}
}

#endif