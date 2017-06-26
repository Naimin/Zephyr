#ifndef VERTEX_H
#define VERTEX_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Common
	{
		struct ZEPHYR_COMMON_API Vertex
		{
			Vertex(float x = -1.0f, float y = -1.0f, float z = -1.0f, float nx = -1.0f, float ny = -1.0f, float nz = -1.0f, float u = -1.0f, float v = -1.0f);

			bool operator==(const Vertex& rhs);
			bool operator!=(const Vertex& rhs);
			bool operator<(const Vertex& rhs);

			Vector4f pos;
			Vector4f color;
			Vector3f normal;
			Vector2f uv;
		};

		ZEPHYR_COMMON_API bool operator<(const Vertex& lhs, const Vertex& rhs);
		ZEPHYR_COMMON_API bool operator==(const Vertex& lhs, const Vertex& rhs);
	}
}

#endif
