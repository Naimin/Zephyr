#ifndef OM_MESH_H
#define OM_MESH_H

#include "stdfx.h"

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
				OpenMeshMesh(Graphics::Mesh &mesh);
				virtual ~OpenMeshMesh();

			protected:
				OMMesh mMesh;
		};
	}
}

#endif