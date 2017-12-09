#ifndef MESH_CONVERTER_H
#define MESH_CONVERTER_H

#include "stdfx.h"

#include "Mesh/Model.h"

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API MeshLoader
		{
			public:
				MeshLoader();
				virtual ~MeshLoader();

				static bool loadFile(const std::string& path, Model* pModel);
		};
	}
}

#endif
