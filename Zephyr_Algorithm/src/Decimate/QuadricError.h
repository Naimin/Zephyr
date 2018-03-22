#ifndef ZEPHYR_ALGORITHM_QUADRIC_ERROR_H
#define ZEPHYR_ALGORITHM_QUADRIC_ERROR_H

#include "../stdfx.h"
#include <Mesh/OM_Mesh.h>
#include <OpenMesh/Core/Geometry/QuadricT.hh>
#include <OpenMesh/Tools/Decimater/CollapseInfoT.hh>

#include <vector>

/// Quadric using double
typedef OpenMesh::Geometry::QuadricT<double> QuadricD;
typedef OpenMesh::Decimater::CollapseInfoT<Zephyr::Common::OMMesh> CollapseInfo;

namespace Zephyr
{
	namespace Algorithm
	{
		const float INVALID_COLLAPSE = 10000.0f; // super big error, so the std::map always place it last

		class ZEPHYR_ALGORITHM_API QuadricError
		{
			public:
				static float computeQuadricError(HalfedgeHandle halfEdgeHandle, Common::OpenMeshMesh& mesh, float maxError, float maxAngle);

				static QuadricD computeQuadricForFace(FaceHandle faceHandle, Common::OpenMeshMesh& mesh);
				static QuadricD computeQuadricForVertex(VertexHandle vertexHandle, Common::OpenMeshMesh& mesh);

				static float computeTriangleFlipAngle(CollapseInfo& collapseInfo, Common::OpenMeshMesh& mesh, float maxAngle);
				static float computeTriangleFlipAngle(CollapseInfo & collapseInfo, Common::OMMesh & omesh, float maxAngle);

			private:
				float mMaxError;
		};

	}
}

#endif