#include "MeshConverter.h"
#include <tbb/parallel_for.h>
#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

using namespace Zephyr::Common;

MeshConverter::MeshConverter()
{
}

MeshConverter::~MeshConverter()
{
}

OpenMeshMesh Zephyr::Common::MeshConverter::ModelToOpenMesh(const Model & model)
{
	return OpenMeshMesh();
}

