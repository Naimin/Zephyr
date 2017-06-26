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
				Material(boost::filesystem::path path = boost::filesystem::path(""), std::string name = std::string());
				virtual ~Material();

				boost::filesystem::path& getPath();

				std::string mName;
				boost::filesystem::path mPath;
				Vector3f mAmbientColor;
				Vector3f mEmissiveColor;
				Vector3f mDiffuseColor;
				Vector3f mSpecularColor;
		};
	}
}

#endif
