#include "Model.h"

using namespace Zephyr::Common;

Model::Model()
{

}

Model::~Model()
{

}

void Model::addMesh(Mesh& mesh)
{
	mMeshes.push_back(mesh);
}

Mesh& Model::getMesh(const int i)
{
	return mMeshes[i];
}

const Mesh& Model::getMesh(const int i) const
{
	return mMeshes[i];
}

std::vector<Mesh>& Model::getMeshes()
{
	return mMeshes;
}

const std::vector<Mesh>& Model::getMeshes() const
{
	return mMeshes;
}

int Model::getMeshesCount() const
{
	return int(mMeshes.size());
}

void Model::addMaterial(Material& material)
{
	mMaterials.push_back(material);
}

Material& Model::getMaterial(const int i)
{
	return mMaterials[i];
}

const Material& Model::getMaterial(const int i) const
{
	return mMaterials[i];
}

std::vector<Material>& Model::getMaterials()
{
	return mMaterials;
}

const std::vector<Material>& Model::getMaterials() const
{
	return mMaterials;
}

int Model::getMaterialsCount() const
{
	return int(mMaterials.size());
}

void Model::addTexture(const boost::filesystem::path& texturePath)
{
	mTextures.push_back(Texture(texturePath));
}

Texture& Model::getTexture(const int i)
{
	return mTextures[i];
}

const Texture& Model::getTexture(const int i) const
{
	return mTextures[i];
}

std::vector<Texture>& Model::getTextures()
{
	return mTextures;
}

const std::vector<Texture>& Model::getTextures() const
{
	return mTextures;
}

int Model::getTexturesCount() const
{
	return int(mTextures.size());
}