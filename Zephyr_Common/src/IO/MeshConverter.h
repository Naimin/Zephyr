#ifndef MESH_CONVERTER_H
#define MESH_CONVERTER_H

#include "../stdfx.h"

#include "../Mesh/Model.h"

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API MeshConverter
		{
			public:
				MeshConverter();
				virtual ~MeshConverter();

				//static bool loadFile(const std::string& path, Model* pModel);*/
		};
	}
}

#endif
