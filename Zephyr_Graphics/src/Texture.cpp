#include "Texture.h"
#include "Zephyr_Graphics.h"

Zephyr::Graphics::Texture::Texture(const std::string & path) : mPath(path), bValid(false)
{
	bValid = loadFromFile(path);
}

Zephyr::Graphics::Texture::~Texture()
{
}

bool Zephyr::Graphics::Texture::isValid()
{
	return bValid;
}

std::string Zephyr::Graphics::Texture::getPath()
{
	return mPath;
}

fipImage Zephyr::Graphics::Texture::getRawData()
{
	return mFreeImageData;
}

bool Zephyr::Graphics::Texture::loadFromFile(const std::string & path)
{
	auto success = mFreeImageData.load(path.c_str());
	if (!success)
		return false;

	success = mFreeImageData.convertTo32Bits();
	if (!success)
		return false;
	
	

	/*success = mFreeImageData.flipVertical();
	if (!success)
		return false;
		*/
	return true;
}
