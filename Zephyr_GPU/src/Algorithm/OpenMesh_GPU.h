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

		struct QEM_Data
		{
			std::vector<Common::Vector3f> vertices; // vertex buffer
			std::vector<unsigned char> indices; // all the faces formed
		};

		struct OpenMesh_GPU
		{
			std::vector<QEM_Data> copyPartialMesh(Common::OMMesh& mesh, const std::vector<int>& randomList);
		
		protected:
			void collectOneRingNeighbour(VertexHandle vh, Common::OMMesh & mesh, std::vector<Common::Vector3f>& vertexBuffer, std::vector<INDEX_TYPE>& indexBuffer, std::map<SortVector3f, INDEX_TYPE>& uniqueVertex);
		};
	}
}

#endif