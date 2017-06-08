#ifndef MESH_H
#define MESH_H

#include "stdfx.h"
#include "Vertex.h"

namespace Zephyr
{
	namespace Graphics
	{
		class ZEPHYR_GRAPHICS_API Mesh
		{
			public:
				std::vector<Zephyr::Vertex>& getVertices();
				const std::vector<Zephyr::Vertex>& getVertices() const;
				std::vector<unsigned int>& getIndices();
				const std::vector<unsigned int>& getIndices() const;
				int getFaceCount() const;

				int mMaterialId;
				std::vector<Vertex> mVertices;
				std::vector<unsigned int> mIndices;
		};
	}
}

#endif
