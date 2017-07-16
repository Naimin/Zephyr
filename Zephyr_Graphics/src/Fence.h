#ifndef FENCE_H
#define FENCE_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Graphics
	{
		class CommandQueue;

		class ZEPHYR_GRAPHICS_API Fence
		{
			public:
				Fence(SharedPtr<ID3D12Device> pDevice);
				virtual ~Fence();

				UINT64 current(); // get current value;
				void increment(); // increment fence value
				void signal(); // CPU signal 
				bool signal(SharedPtr<ID3D12CommandQueue> pCommandQueue); // GPU signal

				// non-blocking check
				bool isDone();

				// blocking
				void waitForFence();
				void waitAndIncrement();

			public:
				ID3D12Fence* mFence;   		 
				UINT64 mFenceValue;
				HANDLE mFenceEvent;
		};

	}
}

#endif
