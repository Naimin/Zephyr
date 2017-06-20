#include "Material.h"

using namespace Zephyr::Common;

Material::Material(const boost::filesystem::path& path) : mPath(path)
{

}

Material::~Material()
{

}

boost::filesystem::path& Material::getPath()
{
	return mPath;
}