#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include "stdfx.h"
#include "CommandQueue.h"

namespace Zephyr
{
	namespace Graphics
	{
		class GraphicsEngine;

		class ZEPHYR_GRAPHICS_API ResourceManager
		{
			public:
				enum RESOURCE_TYPE
				{
					DEFAULT = 1, // GPU read/write, no CPU
					UPLOAD, // GPU read, CPU write
					READ_BACK, // GPU write, CPU read
				};

			public:
				ResourceManager(GraphicsEngine* pGraphicsEngine);
				virtual ~ResourceManager();
				bool initialize();

				SharedPtr<ID3D12Resource> createResource(const std::wstring & resourceName, int bufferSize, RESOURCE_TYPE type, D3D12_RESOURCE_DESC* description = nullptr);
				SharedPtr<ID3D12Resource> createDepthStencilResource(const std::wstring & resourceName, D3D12_CLEAR_VALUE& clearValue);
				bool copyDataToResource(const std::wstring& toResourceName, const std::wstring& fromResourceName, int bufferSize, void* pData, int rowPitch = -1, int height = -1);
				bool createAndCopyToGPU(const std::wstring& resourceName, int bufferSize, void* pData, int rowPitch = -1, int height = -1);
				bool createTextureAndCopyToGPU(const std::wstring& resourceName, D3D12_RESOURCE_DESC* description, int bufferSize, void* pData, int rowPitch = -1, int height = -1);
				bool copyTextureToResource(const std::wstring& toResourceName, const std::wstring& fromResourceName, int bufferSize, void* pData, int rowPitch = -1, int height = -1);

				SharedPtr<ID3D12Resource> getResource(const std::wstring& resourceName);
				void releaseResource(const std::wstring& resourceName);

				void waitForPreviousTask();
				D3D12_VERTEX_BUFFER_VIEW getVertexResourceView(const std::wstring & resourceName, const int stride);
				D3D12_INDEX_BUFFER_VIEW getIndexResourceView(const std::wstring & resourceName);
				
				GraphicsEngine* getEngine();

			protected:
				bool createCommandQueue();

			private:
				GraphicsEngine* mpGraphicsEngine;
				std::unordered_map<std::wstring, SharedPtr<ID3D12Resource>> mResources;
				std::unordered_map<std::wstring, int> mResourceSize;

				std::shared_ptr<CommandQueue> mpCommandQueue; // container for command lists
				std::map<std::string, CommandList> mCommands; // command to copy resources
				Fence mFence;
		};
	}
}


#endif
