#ifndef COMMAND_LIST_H
#define COMMAND_LIST_H

#include "stdfx.h"
#include "Fence.h"

namespace Zephyr
{
	namespace Graphics
	{
		class GraphicsEngine;
		class CommandQueue;
		class Pipeline;
		// wrapper over directx commandList
		class CommandList
		{
			public:	
				CommandList(const int frameBufferCount, GraphicsEngine* pEngine);
				virtual ~CommandList();

				virtual void update(const int frameIndex);

				virtual bool startRecording(const int frameIndex);
				virtual bool endRecording(const int frameIndex);

				void signal(const int frameIndex, SharedPtr<ID3D12CommandQueue> pCommandQueue); // allow GPU to signal

				bool ready(); // check if the command list is ready to execute

			public:
				ID3D12GraphicsCommandList* getCommandList();
				ID3D12PipelineState* getPipelineState() const;

			protected:
				bool createCommandAllocator(const int frameBufferCount);
				bool createCommandList();
				bool createFence(const int frameBufferCount);

			protected:
				int mFrameBufferCount;
				bool bClosed;
				GraphicsEngine* mpEngine;
				SharedPtr<ID3D12Device> mpDevice;
				std::vector<ID3D12CommandAllocator*> mCommandAllocators; // we want enough allocators for each buffer * number of threads (we only have one thread)
				SharedPtr<ID3D12GraphicsCommandList> mpCommandList; // list of all available command list we can record commands into, then execute them to render the frame
				
				std::shared_ptr<Pipeline> mpPipelineState; // pso containing a pipeline state

				// for synchronization
				std::vector<std::shared_ptr<Fence>> mFences;
		};

	}
}

#endif
