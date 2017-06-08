#include "Triangle.h"

using namespace Zephyr::Common;

Zephyr::Common::Triangle::Triangle(Point p0, Point p1, Point p2)
{
	mVertex[0] = p0;
	mVertex[1] = p1;
	mVertex[2] = p2;
}

Point Zephyr::Common::Triangle::getVertex(const int i) const
{
	if (i > 2)
		return Point(-1.0f,-1.0f,-1.0f);

	return mVertex[i];
}

Vector3f Zephyr::Common::Triangle::computeNormal() const
{
	auto e1 = mVertex[1].position - mVertex[0].position;
	auto e2 = mVertex[2].position - mVertex[0].position;
	auto cross = e1.cross(e2);

	return cross;
}

Vector3f Zephyr::Common::Triangle::computeNormalNorm() const
{
	auto normal = computeNormal();
	return normal.normalized();
}

float Zephyr::Common::Triangle::computeArea() const
{
	auto cross = computeNormal();
	return 0.5f * cross.norm();
}
