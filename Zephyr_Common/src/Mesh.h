#ifndef MESH_H
#define MESH_H

#include "Vertex.h"

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API Mesh
		{
			public:
				Mesh();
				virtual ~Mesh();

				std::vector<Vertex>& getVertices();
				const std::vector<Vertex>& getVertices() const;

				int getVerticesCount() const;
				void resizeVertices(const int size);

				std::vector<int>& getIndices();
				const std::vector<int>& getIndices() const;

				int getIndicesCount() const;
				void resizeIndices(const int size);

				int getFaceCount() const;
				
				int getMaterialId() const;
				void setMaterialId(const int id);

			protected:
				std::vector<Vertex> mVertices;
				std::vector<int> mIndices;
				int mMaterialId;
		};
	}
}

#endif
