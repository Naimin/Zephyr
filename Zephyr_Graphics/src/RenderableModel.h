#ifndef RENDERABLE_MODEL_H
#define RENDERABLE_MODEL_H

#include "stdfx.h"

#include <Mesh/Model.h>
#include <Mesh/Material.h>

namespace Zephyr
{
	namespace Graphics
	{
		class CommandList;
		class ResourceManager;
		class OpenMeshMesh;
		// class of renderable model
		class ZEPHYR_GRAPHICS_API RenderableModel : public Common::Model
		{
			public:
				// HACK
					RenderableModel(const std::wstring& name) : mpResourceManager(nullptr) {}
				// HACK end
				RenderableModel(const std::wstring& name, ResourceManager* pResourceManager);
				virtual ~RenderableModel();

				D3D12_VERTEX_BUFFER_VIEW getVertexResourceView(const int meshId);
				D3D12_INDEX_BUFFER_VIEW getIndexResourceView(const int meshId);

				bool loadFromFile(const std::string& path);
				bool loadOpenMesh(const OpenMeshMesh & mesh);
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
				std::vector<ID3D12DescriptorHeap*> mTextureHeap;
		};

	}
}

#endif
