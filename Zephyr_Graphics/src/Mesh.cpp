#include "Mesh.h"

std::vector<Zephyr::Vertex>& Zephyr::Graphics::Mesh::getVertices()
{
	return mVertices;
}

const std::vector<Zephyr::Vertex>& Zephyr::Graphics::Mesh::getVertices() const
{
	return mVertices;
}

std::vector<unsigned int>& Zephyr::Graphics::Mesh::getIndices()
{
	return mIndices;
}

const std::vector<unsigned int>& Zephyr::Graphics::Mesh::getIndices() const
{
	return mIndices;
}

int Zephyr::Graphics::Mesh::getFaceCount() const
{
	return (int)mIndices.size() / 3;
}
