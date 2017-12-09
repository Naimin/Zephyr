#include "Material.h"

using namespace Zephyr::Common;

Material::Material(boost::filesystem::path path, std::string name) 
	: mPath(path), mName(name), 
	mAmbientColor(Vector3f(0,0,0)),
	mEmissiveColor(Vector3f(0,0,0)),
	mDiffuseColor(Vector3f(0,0,0)),
	mSpecularColor(Vector3f(0,0,0))
{

}

Material::~Material()
{

}

boost::filesystem::path& Material::getPath()
{
	return mPath;
}