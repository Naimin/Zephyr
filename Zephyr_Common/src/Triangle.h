#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"

namespace Zephyr
{
	namespace Common
	{
		struct Triangle
		{
			Triangle(const Point p0 = Point(), const Point p1 = Point(), const Point p2 = Point());

			Vector3f getNormal() const;
			Point getVertex(const int i) const;
			float computeArea() const;

			Point mVertex[3];
		};
	}
}

#endif