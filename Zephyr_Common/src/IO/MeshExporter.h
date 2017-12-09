#ifndef MESH_EXPORTER_H
#define MESH_EXPORTER_H

#include "../stdfx.h"

#include "../Mesh/Model.h"

namespace Zephyr
{
	namespace Common
	{

		class ZEPHYR_COMMON_API MeshExporter
		{
			public:
				MeshExporter();
				virtual ~MeshExporter();

				static bool exportMesh(const std::string& path, Model* pModel);
		};

	}
}

#endif
