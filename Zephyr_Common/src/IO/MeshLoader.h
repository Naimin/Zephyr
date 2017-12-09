#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include "../stdfx.h"

#include "../Mesh/Model.h"

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
