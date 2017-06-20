#include "Material.h"

Zephyr::Common::Material::Material(const boost::filesystem::path& path) : mPath(path), mDiffuseTex(path)
{

}

Zephyr::Common::Material::~Material()
{

}

Zephyr::Common::Texture& Zephyr::Common::Material::getDiffuseTexture()
{
	return mDiffuseTex;
}