#ifndef TEXTURE_H
#define TEXTURE_H

#include "stdfx.h"
#include <FreeImagePlus.h>
#include <boost/filesystem/path.hpp>

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API Texture
		{
			public:
				Texture(const boost::filesystem::path& path);
				virtual ~Texture();

				bool isValid() const;
				boost::filesystem::path getPath() const;
				fipImage getRawData();

			protected:
				bool loadFromFile(const boost::filesystem::path& path);

			private:
				bool bValid;
				boost::filesystem::path mPath;
				fipImage mFreeImageData;
		};
	}
}

#endif
