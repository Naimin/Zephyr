#ifndef VERTEX_H
#define VERTEX_H

#include "stdfx.h"

namespace Zephyr
{
	struct ZEPHYR_GRAPHICS_API Vertex
	{
		Vertex();
		Vertex(float x, float y, float z, float nx, float ny, float nz, float u, float v);

		DirectX::XMFLOAT4 pos;
		DirectX::XMFLOAT3 normal;
		DirectX::XMFLOAT4 color;
		DirectX::XMFLOAT2 uv;
	};
}


#endif
