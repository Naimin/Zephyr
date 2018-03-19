#ifndef ZEPHYR_GPU_ALGORITHM_DECIMATE_H
#define ZEPHYR_GPU_ALGORITHM_DECIMATE_H

#include "../stdfx.h"
#include <Mesh/OM_Mesh.h>

namespace Zephyr
{
	namespace GPU
	{
		int ZEPHYR_GPU_API decimate(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize = 8);
	}
}

#endif