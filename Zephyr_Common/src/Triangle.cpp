#include "Triangle.h"

using namespace Zephyr::Common;

Zephyr::Common::Triangle::Triangle(const Point p0, const Point p1, const Point p2)
{
	mVertex[0] = p0;
	mVertex[1] = p1;
	mVertex[2] = p2;
}

Vector3f Zephyr::Common::Triangle::getNormal() const
{
	return Vector3f();
}

Point Zephyr::Common::Triangle::getVertex(const int i) const
{
	if (i > 2)
		return Point(-1.0f,-1.0f,-1.0f);

	return mVertex[i];
}

float Zephyr::Common::Triangle::computeArea() const
{
	auto e1 = mVertex[0].position - mVertex[1].position;
	auto e2 = mVertex[0].position - mVertex[2].position;
	auto cross = e1.cross(e2);

	return 0.5f * cross.norm();
}
