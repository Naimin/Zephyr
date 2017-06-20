#ifndef VERTEX_H
#define VERTEX_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Common
	{
		struct ZEPHYR_COMMON_API Vertex
		{
			Vertex();
			Vertex(float x, float y, float z, float nx, float ny, float nz, float u, float v);

			Vector4f pos;
			Vector4f color;
			Vector3f normal;
			Vector2f uv;
		};
	}
}

#endif
