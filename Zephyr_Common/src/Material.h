#ifndef MATERIAL_H
#define MATERIAL_H

#include "stdfx.h"
#include "Texture.h"

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API Material
		{
			public:
				Material(const boost::filesystem::path& path);
				virtual ~Material();

				boost::filesystem::path& getPath();

			protected:
				boost::filesystem::path mPath;
		};
	}
}

#endif
