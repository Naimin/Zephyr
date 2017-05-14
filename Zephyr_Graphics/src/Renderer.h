#ifndef RENDERER_H
#define RENDERER_H

#include "stdfx.h"
#include "IRenderPass.h"
#include "CommandQueue.h"

namespace Zephyr
{
	namespace Graphics
	{
		const int FRAME_BUFFER_COUNT = 3; // triple buffering
		
		class GraphicsEngine;

		class ZEPHYR_GRAPHICS_API Renderer
		{
			public:
				Renderer(GraphicsEngine* pEngine);
				virtual ~Renderer();

				// function declarations
				bool initialize(unsigned int backBufferWidth, unsigned int backBufferHeight, HWND& hwnd, bool bFullscreen); // initializes direct3d 12

				// Render Pass management
				void addRenderPass(const std::string& renderPassName, IRenderPass* pRenderPass); // update the direct3d pipeline (update command lists)
				// automatically create new command list when queue index is higher current command list count
				bool enqueuRenderPass(const std::string& renderPassName, const int queueIndex);
				void clearRenderPassQueue(const int queueIndex);

				void render(); // execute the command list

				void waitForPreviousFrame(); // wait until gpu is finished with command list

			public: // accessor
				bool isRunning() const;
				SharedPtr<ID3D12Device> getDevice() const;
				SharedPtr<ID3D12DescriptorHeap> getDescriptionHeap() const;
				int getFrameIndex() const;
				int getRtvDescriptorSize() const;
				//SharedPtr<ID3D12CommandQueue> getCommandQueue() const;
				//std::shared_ptr<CommandList> getCommandList(const int Id) const;
				//int getCommandListCount() const;
				std::vector<ID3D12Resource*> getRenderTargets() const;
				DXGI_SAMPLE_DESC getSampleDesc() const;

			protected:
				bool createDevice();
				bool createCommandQueue();
				bool createSwapChain(HWND& hwnd, bool bFullScreen);
				bool createRenderTargetView();
				bool createFence();

				void update(int commandListId);

				void cleanup(); // release com ojects and clean up memory

			private:
				GraphicsEngine* mpEngine;
				bool bIsRunning;
				unsigned int mBackBufferWidth, mBackBufferHeight;

				SharedPtr<IDXGIFactory4> mpDxgiFactory;

				SharedPtr<ID3D12Device> mpDevice; // direct3d device
				SharedPtr<IDXGISwapChain3> mpSwapChain; // swapchain used to switch between render targets

				SharedPtr<ID3D12DescriptorHeap> mpRtvDescriptorHeap; // a descriptor heap to hold resources like the render targets

				std::vector<ID3D12Resource*> mRenderTargets; // number of render targets equal to buffer count

				std::vector<std::shared_ptr<CommandQueue>> mCommandQueues; // 

				int mFrameIndex; // current rtv we are on

				int mRtvDescriptorSize; // size of the rtv descriptor on the device (all front and back buffers will be the same size)

				std::vector<std::shared_ptr<Fence>> mFences;
				// 
				std::unordered_map <std::string, std::shared_ptr<IRenderPass>> mRenderPassMap; // storage for Render Pass

				DXGI_SAMPLE_DESC mSampleDesc;
		};
	}
}

#endif