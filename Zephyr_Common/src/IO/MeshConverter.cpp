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
	OpenMeshMesh omesh;

	// for each mesh in the model add it into the open mesh
	auto meshes = model.getMeshes();
	for (auto mesh : meshes)
	{
		omesh.loadMesh(mesh);
	}

	return omesh;
}

Model Zephyr::Common::MeshConverter::OpenMeshToModel(const OpenMeshMesh & omesh)
{
	return Model();
}

