#ifndef OM_MESH_H
#define OM_MESH_H

#include "../stdfx.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

// Wrapper class foor OpenMesh half-edge mesh representation
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>

#include "Mesh.h"

using namespace OpenMesh;

struct ZephyrTraits : public OpenMesh::DefaultTraits
{
	VertexAttributes(OpenMesh::Attributes::Status);
	FaceAttributes(OpenMesh::Attributes::Status|OpenMesh::Attributes::Normal);
	EdgeAttributes(OpenMesh::Attributes::Status);
	HalfedgeAttributes(OpenMesh::Attributes::Status);
};

namespace Zephyr
{
	namespace Common
	{
		// define traits
		typedef OpenMesh::TriMesh_ArrayKernelT<ZephyrTraits> OMMesh;

		class ZEPHYR_COMMON_API OpenMeshMesh
		{
			public:
				OpenMeshMesh();
				// take in a Zephyr::Graphics::Mesh and build half edge
				OpenMeshMesh(Common::Mesh &mesh);

				OpenMeshMesh(const std::string& path);

				virtual ~OpenMeshMesh();

				void loadMesh(Common::Mesh& mesh);

				OMMesh& getMesh();

				bool exports(const std::string& path);

			protected:
				OMMesh mMesh;
		};
	}
}

#endif