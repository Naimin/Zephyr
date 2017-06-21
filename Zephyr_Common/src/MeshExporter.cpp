#include "MeshExporter.h"

#include <assimp/Exporter.hpp>
#include <assimp/scene.h>

#include <tbb/parallel_for.h>

#include <iostream>

#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string.hpp>

Zephyr::Common::MeshExporter::MeshExporter()
{
}

Zephyr::Common::MeshExporter::~MeshExporter()
{
}

bool Zephyr::Common::MeshExporter::exportMesh(const std::string & path, Model * pModel)
{
	aiScene scene;

	// create the material
	int materialCount = pModel->getMaterialsCount();
	scene.mMaterials = new aiMaterial*[materialCount];
	scene.mNumMaterials = materialCount;
	for (unsigned int materialId = 0; materialId < materialCount; ++materialId)
	{
		aiMaterial* pMaterial = new aiMaterial();

		scene.mMaterials[materialId] = pMaterial;
	}

	int meshesCount = pModel->getMeshesCount();
	scene.mMeshes = new aiMesh*[meshesCount];
	scene.mNumMeshes = meshesCount;

	// create a root node
	scene.mRootNode = new aiNode();
	scene.mRootNode->mMeshes = new unsigned int[meshesCount];
	for (int i = 0; i < meshesCount; ++i)
	{
		scene.mRootNode->mMeshes[i] = i;
	}

	scene.mRootNode->mNumMeshes = meshesCount;

	for (unsigned int meshId = 0; meshId < scene.mNumMeshes; ++meshId)
	{
		aiMesh* pMesh = new aiMesh();
		Mesh& mesh = pModel->getMesh(meshId);
		
		int verticesCount = mesh.getVerticesCount();
		pMesh->mMaterialIndex = mesh.getMaterialId();
		// vertex buffer
		pMesh->mVertices = new aiVector3D[verticesCount];
		pMesh->mNumVertices = verticesCount;
		// texture coordinate buffer
		pMesh->mTextureCoords[0] = new aiVector3D[verticesCount];
		pMesh->mNumUVComponents[0] = verticesCount;

		auto vertices = mesh.getVertices();
		tbb::parallel_for(0, verticesCount, [&](const int i)
		{
			auto vertex = vertices[i];
			pMesh->mVertices[i] = aiVector3D(vertex.pos.x(), vertex.pos.y(), vertex.pos.z());
			pMesh->mTextureCoords[0][i] = aiVector3D(vertex.uv.x(), vertex.uv.y(), 0);
		});

		// face / index buffer
		int faceCount = mesh.getFaceCount();
		pMesh->mFaces = new aiFace[faceCount];
		pMesh->mNumFaces = faceCount;

		auto indices = mesh.getIndices();

		tbb::parallel_for(0, faceCount, [&](const int i)
		{
			aiFace& face = pMesh->mFaces[i];

			face.mIndices = new unsigned int[3];
			face.mNumIndices = 3;

			int index = i * 3;
			face.mIndices[0] = indices[index];
			face.mIndices[1] = indices[index+1];
			face.mIndices[2] = indices[index+2];
		});

		scene.mMeshes[meshId] = pMesh;
	}

	Assimp::Exporter *exporter = new Assimp::Exporter();
	
	// Find the exporter that the user requested
	auto exportFormatDescCount = exporter->GetExportFormatCount();
	
	boost::filesystem::path outpath = path;
	auto fileExt = outpath.extension().string();
	fileExt.erase(0, 1); // remove the dot

	const aiExportFormatDesc* pExportFormatDesc = nullptr;
	for (int i = 0; i < exportFormatDescCount; ++i)
	{
		auto exportFormatDesc = exporter->GetExportFormatDescription(i);
		if (boost::iequals(exportFormatDesc->fileExtension, fileExt))
		{
			pExportFormatDesc = exportFormatDesc;
			break;
		}
	}

	if (nullptr == pExportFormatDesc)
	{
		std::cout << "Failed to find " << fileExt << " Exporter." << std::endl;
		return false;
	}

	if (aiReturn_SUCCESS != exporter->Export(&scene, pExportFormatDesc->id, outpath.string()))
	{
		std::cout << "Failed to export : " << path << std::endl;
		return false;
	}
	
	return true;
}
