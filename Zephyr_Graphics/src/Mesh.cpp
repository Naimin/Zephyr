#include "Mesh.h"

std::vector<Zephyr::Vertex>& Zephyr::Graphics::Mesh::getVertices()
{
	return mVertices;
}

std::vector<unsigned int>& Zephyr::Graphics::Mesh::getIndices()
{
	return mIndices;
}

int Zephyr::Graphics::Mesh::getFaceCount() const
{
	return (int)mIndices.size() / 3;
}
