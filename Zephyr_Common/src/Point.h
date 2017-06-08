#ifndef POINT_H
#define POINT_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Common
	{
		struct ZEPHYR_COMMON_API Point
		{
			Point(float x = 0, float y = 0, float z = 0) : position(x,y,z)
			{

			}

			Vector3f position;
		};
	}
}

#endif
