#ifndef RENDERABLE_MODEL_H
#define RENDERABLE_MODEL_H

#include "stdfx.h"

#include "Mesh.h"
#include "Material.h"

namespace Zephyr
{
	namespace Graphics
	{
		class CommandList;
		class ResourceManager;
		// class of renderable model
		class ZEPHYR_GRAPHICS_API RenderableModel
		{
			public:
				// HACK
					RenderableModel(const std::wstring& name) : mpResourceManager(nullptr) {}
				// HACK end
				RenderableModel(const std::wstring& name, ResourceManager* pResourceManager);
				virtual ~RenderableModel();

				void addMesh(Mesh& mesh);
				Mesh& getMesh(const int meshId);
				int getMeshCount() const;

				void addMaterial(Material& material);
				Material& getMaterial(const int materialId);
				int getMaterialCount() const;

				D3D12_VERTEX_BUFFER_VIEW getVertexResourceView(const int meshId);
				D3D12_INDEX_BUFFER_VIEW getIndexResourceView(const int meshId);

				bool loadFromFile(const std::string& path);
				bool uploadToGPU();
				void unloadFromGPU();

				bool drawMesh(const int meshId, SharedPtr<ID3D12GraphicsCommandList> pCommandList);

			protected:
				std::wstring getVertexResourceName(const int meshId) const;
				std::wstring getIndexResourceName(const int meshId) const;

				bool uploadVerticesToGPU(const int meshId);
				bool uploadIndicesToGPU(const int meshId);
				bool uploadTextureToGPU(const int materialId);

			private:
				ResourceManager* mpResourceManager;

				std::wstring mPath;
				std::wstring mName;

				// each material will have a mesh
				std::vector<Material> mMaterials;
				std::vector<Mesh> mMeshes;
				std::vector<Texture> mTextures;
				std::vector<ID3D12DescriptorHeap*> mTextureHeap;
		};

	}
}

#endif
