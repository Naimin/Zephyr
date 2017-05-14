#include "Vertex.h"


Zephyr::Vertex::Vertex()
{
}

Zephyr::Vertex::Vertex(float x, float y, float z, float nx, float ny, float nz, float u, float v) : pos(x, y, z, 1.0f), normal(nx, ny, nz), color(1.0f, 1.0f, 1.0f, 1.0f), uv(u, v)
{
}