#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Point.h"
#include "Vertex.h"

namespace Zephyr
{
	namespace Common
	{
		struct ZEPHYR_COMMON_API Triangle
		{
		public:
			Triangle(Vertex p0 = Vertex(), Vertex p1 = Vertex(), Vertex p2 = Vertex());

			Vertex getVertex(const int i) const;
			Vector3f computeNormal() const;
			Vector3f computeNormalNorm() const;
			float computeArea() const;

			Vertex mVertex[3];
		};
	}
}

#endif
