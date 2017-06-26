#include "Triangle.h"

using namespace Zephyr::Common;

Zephyr::Common::Triangle::Triangle(Vertex p0, Vertex p1, Vertex p2)
{
	mVertex[0] = p0;
	mVertex[1] = p1;
	mVertex[2] = p2;
}

Vertex Zephyr::Common::Triangle::getVertex(const int i) const
{
	if (i > 2)
		return Vertex(-1.0f,-1.0f,-1.0f);

	return mVertex[i];
}

Vector3f Zephyr::Common::Triangle::computeNormal() const
{
	auto v0 = Vector3f(mVertex[0].pos.x(), mVertex[0].pos.y(), mVertex[0].pos.z());
	auto v1 = Vector3f(mVertex[1].pos.x(), mVertex[1].pos.y(), mVertex[1].pos.z());
	auto v2 = Vector3f(mVertex[2].pos.x(), mVertex[2].pos.y(), mVertex[2].pos.z());

	auto e1 = v1 - v0;
	auto e2 = v2 - v0;
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
