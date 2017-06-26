#include "Vertex.h"

using namespace Zephyr::Common;

Vertex::Vertex(float x, float y, float z, float nx, float ny, float nz, float u, float v) :
	pos(x, y, z, 1), normal(nx, ny, nz), uv(u, v)
{

}

bool Vertex::operator==(const Vertex& rhs)
{
	if (this->pos == rhs.pos &&
		this->normal == rhs.normal &&
		this->uv == rhs.uv)
		return true;
	else 
		return false;
}

bool Vertex::operator!=(const Vertex& rhs)
{
	return !(*this == rhs);
}

bool Vertex::operator<(const Vertex& rhs)
{
	if (this->pos.x() > rhs.pos.x() ||
		this->pos.y() > rhs.pos.y() ||
		this->pos.z() > rhs.pos.z() ||
		this->uv.x() > rhs.uv.x() ||
		this->uv.y() > rhs.uv.y() ||
		this->normal.x() > rhs.normal.x() ||
		this->normal.y() > rhs.normal.y() ||
		this->normal.z() > rhs.normal.z() ||
		*this == rhs)
		return false;
	else
		return true;
}

bool Zephyr::Common::operator<(const Vertex& lhs, const Vertex& rhs)
{
	if (lhs.pos.x() != rhs.pos.x())
		return lhs.pos.x() < rhs.pos.x();
	if (lhs.pos.y() != rhs.pos.y())
		return lhs.pos.y() < rhs.pos.y();
	if (lhs.pos.z() != rhs.pos.z())
		return lhs.pos.z() < rhs.pos.z();
	if (lhs.uv.x() != rhs.uv.x())
		return lhs.uv.x() < rhs.uv.x();
	if (lhs.uv.y() != rhs.uv.y())
		return lhs.uv.y() < rhs.uv.y();
	if (lhs.normal.x() != rhs.normal.x())
		return lhs.normal.x() < rhs.normal.x();
	if (lhs.normal.y() != rhs.normal.y())
		return lhs.normal.y() < rhs.normal.y();
	// last test
	return lhs.normal.z() < rhs.normal.z();
}

bool Zephyr::Common::operator==(const Vertex& lhs, const Vertex& rhs)
{
	if (lhs.pos.x() == rhs.pos.x() &&
		lhs.pos.y() == rhs.pos.y() &&
		lhs.pos.z() == rhs.pos.z() &&
		lhs.uv.x() == rhs.uv.x() &&
		lhs.uv.y() == rhs.uv.y() &&
		lhs.normal.x() == rhs.normal.x() &&
		lhs.normal.y() == rhs.normal.y() &&
		lhs.normal.z() == rhs.normal.z())
		return true;
	else
		return false;
}