#ifndef TEXTURE_H
#define TEXTURE_H

#include "stdfx.h"
#include <FreeImagePlus.h>

namespace Zephyr
{
	namespace Graphics
	{
		class GraphicsEngine;

		class Texture
		{
			public:
				Texture(const std::string& path);
				virtual ~Texture();

				bool isValid();
				std::string getPath();
				fipImage getRawData();

			protected:
				bool loadFromFile(const std::string& path);

			private:
				bool bValid;
				std::string mPath;
				fipImage mFreeImageData;
		};

	}
}

#endif
