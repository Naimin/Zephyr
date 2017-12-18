#include "QuadricError.h"

#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>

using namespace Zephyr;
using namespace Zephyr::Common;

typedef Vec3d Vec3;

float Zephyr::Algorithm::QuadricError::computeQuadricError(HalfedgeHandle halfEdgeHandle, OpenMeshMesh& mesh, float maxError, float maxAngle)
{
	CollapseInfo collapseInfo(mesh.getMesh(), halfEdgeHandle);

	if (Algorithm::INVALID_COLLAPSE == computeTriangleFlipAngle(collapseInfo, mesh, maxAngle))
		return Algorithm::INVALID_COLLAPSE;

	QuadricD q = computeQuadricForVertex(collapseInfo.v0, mesh);
	q += computeQuadricForVertex(collapseInfo.v1, mesh);

	// evaluate the quadric error
	double err = q(collapseInfo.p1);

	return float((err < maxError) ? err : Algorithm::INVALID_COLLAPSE);
}

QuadricD Zephyr::Algorithm::QuadricError::computeQuadricForFace(FaceHandle faceHandle, OpenMeshMesh& mesh)
{
	auto& omesh = mesh.getMesh();

	// copied from OpenMesh's ModQuadricT.cc
	auto fv_it = omesh.fv_iter(faceHandle);
	auto vh0 = *fv_it;  ++fv_it;
	auto vh1 = *fv_it;  ++fv_it;
	auto vh2 = *fv_it;

	Vec3 v0, v1, v2;
	{
		using namespace OpenMesh;

		v0 = vector_cast<Vec3>(omesh.point(vh0));
		v1 = vector_cast<Vec3>(omesh.point(vh1));
		v2 = vector_cast<Vec3>(omesh.point(vh2));
	}

	Vec3 n = (v1 - v0) % (v2 - v0);
	double area = n.norm();
	if (area > FLT_MIN)
	{
		n /= area;
		area *= 0.5;
	}

	const double a = n[0];
	const double b = n[1];
	const double c = n[2];
	const double d = -(vector_cast<Vec3>(omesh.point(vh0)) | n);

	QuadricD q(a, b, c, d);
	q *= area;

	return q;
}

QuadricD Zephyr::Algorithm::QuadricError::computeQuadricForVertex(VertexHandle vertexHandle, Common::OpenMeshMesh & mesh)
{
	// iterate over all faee linked to the vertex and compute quadric
	QuadricD q;
	for (OMMesh::VertexFaceIter vf_Itr = mesh.getMesh().vf_iter(vertexHandle); vf_Itr.is_valid(); ++vf_Itr)
	{
		q += computeQuadricForFace(*vf_Itr, mesh);
	}

	return q;
}

float Zephyr::Algorithm::QuadricError::computeTriangleFlipAngle(CollapseInfo & collapseInfo, Common::OpenMeshMesh& mesh, float maxAngle)
{
	auto& omesh = mesh.getMesh();

	// Set the maximum angular deviation of the orignal normal and the new normal in degrees.
	double max_deviation_ = maxAngle / 180.0 * M_PI;
	double min_cos_ = cos(max_deviation_);

	// check for flipping normals
	OMMesh::ConstVertexFaceIter vf_it(omesh, collapseInfo.v0);
	FaceHandle					fh;
	OMMesh::Scalar              c(1.0);

	// simulate collapse
	omesh.set_point(collapseInfo.v0, collapseInfo.p1);

	for (; vf_it.is_valid(); ++vf_it)
	{
		fh = *vf_it;
		if (fh != collapseInfo.fl && fh != collapseInfo.fr)
		{
			OMMesh::Normal n1 = omesh.normal(fh);
			OMMesh::Normal n2 = omesh.calc_face_normal(fh);

			c = dot(n1, n2);

			if (c < min_cos_)
				break;
		}
	}

	// undo simulation changes
	omesh.set_point(collapseInfo.v0, collapseInfo.p0);

	return float((c < min_cos_) ? INVALID_COLLAPSE : c);
}
