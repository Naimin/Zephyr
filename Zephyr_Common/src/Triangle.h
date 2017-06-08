#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"

namespace Zephyr
{
	namespace Common
	{
		struct ZEPHYR_COMMON_API Triangle
		{
			Triangle(Point p0 = Point(), Point p1 = Point(), Point p2 = Point());

			Point getVertex(const int i) const;
			Vector3f computeNormal() const;
			Vector3f computeNormalNorm() const;
			float computeArea() const;

			Point mVertex[3];
		};
	}
}

#endif
