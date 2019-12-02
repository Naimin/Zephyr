#ifndef ZEPHYR_GPU_ALGORITHM_DECIMATE_H
#define ZEPHYR_GPU_ALGORITHM_DECIMATE_H

#include "../stdfx.h"
#include <Decimate/Decimate.h>

namespace Zephyr
{
	namespace GPU
	{
		int ZEPHYR_GPU_API decimate(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize, Algorithm::DecimationType type);

		int ZEPHYR_GPU_API decimateMC(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize);
		int ZEPHYR_GPU_API decimateSuperVertex(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize);


	}
}

#endif