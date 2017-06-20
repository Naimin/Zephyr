#include "MeshLoader.h"
#include <tbb/parallel_for.h>
#include <iostream>

using namespace Zephyr::Common;

MeshLoader::MeshLoader()
{
}

MeshLoader::~MeshLoader()
{
}

bool MeshLoader::loadFile(const std::string & path, Model* pModel)
{
	if (nullptr == pModel)
		return false;

	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType);

	if (!scene)
	{
		std::cout << "Couldn't load model: " << path << std::endl;
		return false;
	}

	for (unsigned int meshId = 0; meshId < scene->mNumMeshes; ++meshId)
	{
		aiMesh* mesh = scene->mMeshes[meshId];

		Mesh modelMesh; // <- our mesh
		modelMesh.setMaterialId(mesh->mMaterialIndex);

		// copy over the vertex information
		int vertexCount = mesh->mNumVertices;
		modelMesh.resizeVertices(vertexCount);
		tbb::parallel_for(0, vertexCount, [&](const int vertexId)
		{
			aiVector3D pos = mesh->mVertices[vertexId];
			aiVector3D normal = mesh->HasNormals() ? mesh->mNormals[vertexId] : aiVector3D(1.0f, 1.0f, 1.0f);
			aiVector3D uv = mesh->HasTextureCoords(0) ? mesh->mTextureCoords[0][vertexId] : aiVector3D(0.0f, 0.0f, 0.0f);

			modelMesh.getVertices()[vertexId] = Vertex(pos.x, pos.y, pos.z, normal.x, normal.y, normal.z, uv.x, uv.y);
		});

		// setup the index buffer
		int faceCount = mesh->mNumFaces;
		modelMesh.resizeIndices(faceCount * 3);
		tbb::parallel_for(0, faceCount, [&](const int faceId)
		{
			const aiFace& face = mesh->mFaces[faceId];
			int currentFaceId = faceId * 3;
			for (int i = 0; i < 3; ++i)
			{
				modelMesh.getIndices()[currentFaceId] = face.mIndices[i];
				++currentFaceId;
			}
		});

		pModel->addMesh(modelMesh);
	}

	int materialCount = scene->mNumMaterials;
	std::map<std::string, int> materialNameToId;
	std::vector<int> materialIndexRemap(materialCount);

	// create a default material
	pModel->addMaterial(Material(""));

	boost::filesystem::path p(path);
	boost::filesystem::path dir = p.parent_path();
	for (int i = 0; i < materialCount; ++i)
	{
		const aiMaterial* material = scene->mMaterials[i];
		int textureId = 0;
		int materialId = 0;
		aiString path;  // filename

		if (material->GetTexture(aiTextureType_DIFFUSE, textureId, &path) == AI_SUCCESS)
		{
			boost::filesystem::path textureFullPath = dir;
			textureFullPath /= path.data;
			std::string textureName = textureFullPath.string();

			auto itr = materialNameToId.find(textureName);
			if (itr == materialNameToId.end()) // texture not found
			{
				materialId = pModel->getMaterialsCount();
				materialNameToId.insert(std::make_pair(textureName, materialId));

				Material modelMaterial(textureName);

				pModel->addMaterial(modelMaterial);
				pModel->addTexture(textureFullPath);
			}
			else // already exist
			{
				materialId = itr->second;
			}

			materialIndexRemap[i] = materialId;
		}
		else
		{
			// point to default material
			materialIndexRemap[i] = 0;
		}
	}

	tbb::parallel_for(0, pModel->getMeshesCount(), [&](const int i)
	{
		auto mesh = pModel->getMesh(i);
		mesh.setMaterialId(materialIndexRemap[mesh.getMaterialId()]);
	});

	return true;
}
