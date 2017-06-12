#ifndef L_NORM_UTIL_H
#define L_NORM_UTIL_H

#include "stdfx.h"
#include <algorithm>
#include <cmath>

namespace Zephyr
{
	namespace Common
	{
		struct ZEPHYR_COMMON_API LNormUtil
		{

			static float L1Norm(const Vector3f vec);

			static float L2Norm(const Vector3f vec);

			static float LInfinityNorm(const Vector3f vec);

		};
	}
}

#endif 
