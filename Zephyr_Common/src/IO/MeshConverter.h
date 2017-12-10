#ifndef MESH_CONVERTER_H
#define MESH_CONVERTER_H

#include "../stdfx.h"

#include "../Mesh/Model.h"
#include "../Mesh/OM_Mesh.h"

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API MeshConverter
		{
			public:
				MeshConverter();
				virtual ~MeshConverter();

				static OpenMeshMesh ModelToOpenMesh(const Model& model);
				static Model OpenMeshToModel(const OpenMeshMesh& omesh);
		};
	}
}

#endif
