#ifndef COMMAND_QUEUE_H
#define COMMAND_QUEUE_H

#include "stdfx.h"
#include "CommandList.h"
#include "IRenderPass.h"

namespace Zephyr
{
	namespace Graphics
	{
		enum COMMAND_QUEUE_TYPE
		{
			DIRECT = 0,
			BUNDLE = 1,
			COMPUTE = 2,
			COPY = 3
		};

		class ZEPHYR_GRAPHICS_API CommandQueue
		{
			public:
				CommandQueue(const int id, COMMAND_QUEUE_TYPE type, SharedPtr<ID3D12Device> pDevice);
				virtual ~CommandQueue();

				bool enqueueCommandList(std::shared_ptr<CommandList> pCommandList);

				void update(const int frameIndex, const double deltaTime);
				bool execute(const int frameIndex);
				void clear();
				void wait();

			public:
				SharedPtr<ID3D12CommandQueue> getCommandQueue() const;
				std::shared_ptr<CommandList> getCommandList(const int index) const;
				SharedPtr<ID3D12Device> getDevice() const;

			protected:
				int mId;
				SharedPtr<ID3D12Device> mpDevice;
				SharedPtr<ID3D12CommandQueue> mpCommandQueue;
				std::vector<std::shared_ptr<CommandList>> mCommandLists; // list of command lists
				Fence mFence;
		};

	}
}

#endif
