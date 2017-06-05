#ifndef POINT_H
#define POINT_H

#include "GeometryMath.h"

namespace Zephyr
{
	namespace Common
	{
		struct Point
		{
			Point(float x = 0, float y = 0, float z = 0) : position(x,y,z)
			{

			}

			Vector3f position;
		};
	}
}

#endif
