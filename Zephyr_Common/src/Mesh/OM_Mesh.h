#ifndef OM_MESH_H
#define OM_MESH_H

#include "../stdfx.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

// Wrapper class foor OpenMesh half-edge mesh representation
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMeshT.hh>

#include "Mesh.h"

using namespace OpenMesh;

namespace Zephyr
{
	namespace Common
	{
		// define traits
		typedef OpenMesh::TriMesh_ArrayKernelT<> OMMesh;

		enum DecimationType
		{
			GREEDY_DECIMATE = 0,
			RANDOM_DECIMATE
		};

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

				int decimate(unsigned int targetFaceCount, DecimationType type = GREEDY_DECIMATE);

				bool exports(const std::string& path);

			protected:
				int decimateGreedy(unsigned int targetFaceCount);

				int decimateRandom(unsigned int targetFaceCount);

			protected:
				OMMesh mMesh;
		};
	}
}

#endif