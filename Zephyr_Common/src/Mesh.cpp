#include "Mesh.h"

using namespace Zephyr::Common;

Mesh::Mesh()
{

}

Mesh::~Mesh()
{

}

std::vector<Vertex>& Mesh::getVertices()
{
	return mVertices;
}

const std::vector<Vertex>& Mesh::getVertices() const
{
	return mVertices;
}

void Mesh::resizeVertices(const int size)
{
	mVertices.resize(size);
}

std::vector<int>& Mesh::getIndices()
{
	return mIndices;
}

const std::vector<int>& Mesh::getIndices() const
{
	return mIndices;
}

void Mesh::resizeIndices(const int size)
{
	mIndices.resize(size);
}

int Mesh::getFaceCount() const
{
	return int(mIndices.size() / 3);
}

int Mesh::getMaterialId() const
{
	return mMaterialId;
}

void Mesh::setMaterialId(const int id)
{
	mMaterialId = id;
}