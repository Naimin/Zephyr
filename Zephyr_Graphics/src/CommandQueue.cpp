#include "CommandQueue.h"
#include <tbb/parallel_for.h>

Zephyr::Graphics::CommandQueue::CommandQueue(const int id, COMMAND_QUEUE_TYPE type, SharedPtr<ID3D12Device> pDevice) : mId(id), mpDevice(pDevice), mFence(pDevice)
{
	if (mpDevice.isNull())
		return;

	// -- Create the Command Queue -- //
	D3D12_COMMAND_QUEUE_DESC cqDesc = {}; // we will be using all the default values
	cqDesc.Type = (D3D12_COMMAND_LIST_TYPE)type;

	ID3D12CommandQueue* pCmdQueue = nullptr;
	auto hr = mpDevice->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&pCmdQueue)); // create the command queue
	if (FAILED(hr) || nullptr == pCmdQueue)
	{
		return;
	}
	mpCommandQueue.reset(pCmdQueue);
}

Zephyr::Graphics::CommandQueue::~CommandQueue()
{
	wait();
	mCommandLists.clear();
}

bool Zephyr::Graphics::CommandQueue::enqueueCommandList(std::shared_ptr<CommandList> pCommandList)
{
	mCommandLists.push_back(pCommandList);
	return true;
}

void Zephyr::Graphics::CommandQueue::update(const int frameIndex)
{
	tbb::parallel_for((size_t)0, mCommandLists.size(), [&](const size_t i)
	{
		auto commandList = mCommandLists[i];
		commandList->update(frameIndex);
	});
}

bool Zephyr::Graphics::CommandQueue::execute(const int frameIndex)
{
	// wait for previous queue to finish
	mFence.waitAndIncrement();

	std::vector<ID3D12CommandList*> ppCommandLists(mCommandLists.size());
	int i = 0;
	for (auto& commandList : mCommandLists)
	{
		if (commandList->ready())
		{
			auto pCom = commandList->getCommandList();
			ppCommandLists[i++] = pCom;
		}
	}

	// execute the array of command lists
	mpCommandQueue->ExecuteCommandLists((UINT)ppCommandLists.size(), &ppCommandLists[0]);

	auto hr = mpDevice->GetDeviceRemovedReason();
	if (FAILED(hr))
	{
		std::cout << mpDevice->GetDeviceRemovedReason() << std::endl;
	}

	// this command goes in at the end of our command queue. we will know when our command queue 
	// has finished because the fence value will be set to "fenceValue" from the GPU since the command
	// queue is being executed on the GPU
	mFence.signal(mpCommandQueue);

	// signal the fence of the command list after they are submitted
	for (auto commandList : mCommandLists)
	{
		commandList->signal(frameIndex, mpCommandQueue);
	}

	return true;
}

void Zephyr::Graphics::CommandQueue::clear()
{
	wait();
	mCommandLists.clear();
}

void Zephyr::Graphics::CommandQueue::wait()
{
	mFence.waitForFence();
}

Zephyr::SharedPtr<ID3D12CommandQueue> Zephyr::Graphics::CommandQueue::getCommandQueue() const
{
	return mpCommandQueue;
}

std::shared_ptr<Zephyr::Graphics::CommandList> Zephyr::Graphics::CommandQueue::getCommandList(const int index) const
{
	if (index >= mCommandLists.size())
		return nullptr;

	return mCommandLists[index];
}

Zephyr::SharedPtr<ID3D12Device> Zephyr::Graphics::CommandQueue::getDevice() const
{
	return mpDevice;
}
