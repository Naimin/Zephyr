#ifndef ZEPHYR_GPU_OpenMesh_H
#define ZEPHYR_GPU_OpenMesh_H

#include "../stdfx.h"
#include <Mesh/OM_Mesh.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

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
		typedef thrust::host_vector<QEM_Data, thrust::system::cuda::experimental::pinned_allocator<QEM_Data>> QEM_Data_List;
		typedef thrust::host_vector<int, thrust::system::cuda::experimental::pinned_allocator<int>> Thrust_host_vector_int;

		struct QEM_Data_Package
		{
			QEM_Data_Package(size_t numQEM_Data);

			QEM_Data_Package(QEM_Data_List& QEM_Datas);

			void setup(QEM_Data_List& QEM_Datas);

			QEM_Data* getDevicePtr();

			~QEM_Data_Package();

			// device ptr
			thrust::device_vector<QEM_Data> mQEM_Data_GPU;
		};

		struct CopyPartialMesh
		{
			CopyPartialMesh(size_t totalSelectionCount);
			QEM_Data* copyPartialMesh(Common::OMMesh& mesh, const Thrust_host_vector_int& randomList);

		protected:
			void collectOneRingNeighbour(VertexHandle vh, Common::OMMesh & mesh, Common::Vector3f* vertexBuffer, INDEX_TYPE& vertexCount, INDEX_TYPE* indexBuffer, INDEX_TYPE& indexCount, std::map<SortVector3f, INDEX_TYPE>& uniqueVertex);

			QEM_Data_List mQEM_Data;
			QEM_Data_Package mQEM_Data_Package;
		};

		

	}
}

#endif