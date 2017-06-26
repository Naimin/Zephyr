#include "Texture.h"

Zephyr::Common::Texture::Texture() : mPath(""), bValid(false)
{

}

Zephyr::Common::Texture::Texture(const boost::filesystem::path& path) : mPath(path), bValid(false)
{
	bValid = loadFromFile(path);
}

Zephyr::Common::Texture::~Texture()
{
}

bool Zephyr::Common::Texture::isValid() const
{
	return bValid;
}

boost::filesystem::path Zephyr::Common::Texture::getPath() const
{
	return mPath;
}

fipImage Zephyr::Common::Texture::getRawData()
{
	return mFreeImageData;
}

bool Zephyr::Common::Texture::loadFromFile(const boost::filesystem::path& path)
{
	auto success = mFreeImageData.load(path.string().c_str());
	if (!success)
		return false;

	success = mFreeImageData.convertTo32Bits();
	if (!success)
		return false;

	return true;
}