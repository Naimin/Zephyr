#include "Vertex.h"

using namespace Zephyr::Common;

Vertex::Vertex()
{

}

Vertex::Vertex(float x, float y, float z, float nx, float ny, float nz, float u, float v) :
	pos(x, y, z, 1), normal(nx, ny, nz), uv(u, v)
{

}
