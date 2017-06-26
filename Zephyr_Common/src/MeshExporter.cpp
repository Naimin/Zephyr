#include "MeshExporter.h"

#include <assimp/Exporter.hpp>
#include <assimp/scene.h>

#include <tbb/parallel_for.h>

#include <iostream>

#include <boost/filesystem.hpp>
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

	// create the materials
	int materialCount = pModel->getMaterialsCount();
	auto materials = pModel->getMaterials();
	scene.mMaterials = new aiMaterial*[materialCount];
	scene.mNumMaterials = materialCount;

	boost::filesystem::path outputPath(path);
	auto outputDirectory = outputPath.parent_path();
	tbb::parallel_for(0, materialCount, [&](const int materialId)
	//for (int materialId = 0; materialId < materialCount; ++materialId)
	{
		aiMaterial* pMaterial = new aiMaterial();
		const auto& material = materials[materialId];
		auto texturePath = material.mPath;
		
		pMaterial->AddProperty(&aiString(material.mName), AI_MATKEY_NAME);

		pMaterial->AddProperty(&(material.mAmbientColor), 1, AI_MATKEY_COLOR_AMBIENT);
		pMaterial->AddProperty(&(material.mEmissiveColor), 1, AI_MATKEY_COLOR_EMISSIVE);
		pMaterial->AddProperty(&(material.mDiffuseColor), 1, AI_MATKEY_COLOR_DIFFUSE);
		pMaterial->AddProperty(&(material.mSpecularColor), 1, AI_MATKEY_COLOR_SPECULAR);

		// copy the texture file to the output directory
		auto textureName = texturePath.filename();
		auto newTexturePath = outputDirectory;
		newTexturePath /= textureName;

		// check source texture exist
		if (boost::filesystem::exists(texturePath))
		{
			// copy from source to destination
			try
			{
				boost::filesystem::copy_file(texturePath, newTexturePath, boost::filesystem::copy_option::overwrite_if_exists);
			}
			catch(std::exception const& e)
			{
				std::cout << e.what() << '\n';
			}
			pMaterial->AddProperty(&aiString(newTexturePath.filename().string()), AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0));
		}

		scene.mMaterials[materialId] = pMaterial;
	});

	// create the meshes
	int meshesCount = pModel->getMeshesCount();
	scene.mMeshes = new aiMesh*[meshesCount];
	scene.mNumMeshes = meshesCount;

	// create a root node
	scene.mRootNode = new aiNode();
	// need to assign the mesh id into the root node in order for them to be found
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

		pMesh->mColors[0] = new aiColor4D[verticesCount];

		auto vertices = mesh.getVertices();
		tbb::parallel_for(0, verticesCount, [&](const int i)
		{
			auto vertex = vertices[i];
			pMesh->mVertices[i] = aiVector3D(vertex.pos.x(), vertex.pos.y(), vertex.pos.z());
			pMesh->mTextureCoords[0][i] = aiVector3D(vertex.uv.x(), vertex.uv.y(), 0);
			pMesh->mColors[0][i] = aiColor4D(vertex.color.x(), vertex.color.y(), vertex.color.z(), vertex.color.w());
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
