#ifndef ZEPHYR_ALGORITHM_DECIMATE_H
#define ZEPHYR_ALGORITHM_DECIMATE_H

#include "../stdfx.h"
#include <Mesh/OM_Mesh.h>

namespace Zephyr
{
	namespace Algorithm
	{
		enum DecimationType
		{
			GREEDY_DECIMATE = 0,
			RANDOM_DECIMATE,
			RANDOM_DECIMATE_VERTEX,
			ADAPTIVE_RANDOM_DECIMATE
		};

		class ZEPHYR_ALGORITHM_API Decimater
		{
			public:
				static int decimate(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, DecimationType type = GREEDY_DECIMATE);

				static int decimateGreedy(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount);
				static int decimateRandom(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize = 8);
				static int decimateRandomVertex(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize = 8);
				static int decimateAdaptiveRandom(Common::OpenMeshMesh& mesh, unsigned int targetFaceCount, unsigned int binSize = 8);
		};

	}
}

#endif