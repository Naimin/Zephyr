#ifndef OM_MESH_H
#define OM_MESH_H

#include "stdfx.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

// Wrapper class foor OpenMesh half-edge mesh representation
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>

#include <Mesh.h>

#include "TriDualGraph.h"

using namespace OpenMesh;

// define traits
typedef OpenMesh::TriMesh_ArrayKernelT<> OMMesh;

namespace Zephyr
{
	namespace Algorithm
	{
		class ZEPHYR_ALGORITHM_API OpenMeshMesh
		{
			public:
				OpenMeshMesh();
				// take in a Zephyr::Graphics::Mesh and build half edge
				OpenMeshMesh(Common::Mesh &mesh);

				OpenMeshMesh(const std::string& path);

				virtual ~OpenMeshMesh();

				OMMesh& getMesh();

			protected:
				OMMesh mMesh;
		};
	}
}

#endif