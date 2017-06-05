#ifndef Line_H
#define Line_H

#include "Point.h"

namespace Zephyr
{
	namespace Common
	{
		struct Line
		{
			Line(const Point& p1, const Point& p2)
			{
				mPoint[0] = p1;
				mPoint[1] = p2;
			}

			Point mPoint[2];
		};
	}
}

#endif
