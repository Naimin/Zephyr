#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stdfx.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "RenderableModel.h"

namespace Zephyr
{
	namespace Graphics
	{

		class ZEPHYR_GRAPHICS_API ModelLoader
		{
			public:
				ModelLoader();
				virtual ~ModelLoader();

				static bool loadFile(const std::string& path, RenderableModel* pModel);
		};

	}
}

#endif
