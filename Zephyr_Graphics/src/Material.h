#ifndef MATERIAL_H
#define MATERIAL_H

#include "stdfx.h"
#include "Texture.h"

namespace Zephyr
{
	namespace Graphics
	{
		class Material
		{
			public:
				Material(std::string diffusePath);
				virtual ~Material();

			public:
				std::string mDiffusePath;
		};
	}
}

#endif
