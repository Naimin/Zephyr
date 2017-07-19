#include "CommandList.h"
#include "CommandQueue.h"
#include "Zephyr_Graphics.h"
#include "Pipeline.h"

Zephyr::Graphics::CommandList::CommandList(const int frameBufferCount, GraphicsEngine* pEngine) : bClosed(false), mpDevice(pEngine->getRenderer()->getDevice()), mpEngine(pEngine), mpPipelineState(new Pipeline(pEngine))
{
	if (mpDevice.isNull())
		return;

	auto success = createCommandAllocator(frameBufferCount);
	if (!success)
		return;

	success = createCommandList();
	if (!success)
		return;

	success = createFence(frameBufferCount);
	if (!success)
		return;
}

Zephyr::Graphics::CommandList::~CommandList()
{
	for (auto allocators : mCommandAllocators)
	{
		SAFE_RELEASE(allocators)
	}
}

void Zephyr::Graphics::CommandList::update(const int frameIndex, const double deltaTime)
{
	startRecording(frameIndex);

	//mpRenderPass->update(frameIndex);

	endRecording(frameIndex);
}

void Zephyr::Graphics::CommandList::signal(const int frameIndex, SharedPtr<ID3D12CommandQueue> pCommandQueue)
{
	// allow the GPU to signal the fence
	mFences[frameIndex]->signal(pCommandQueue);
}

bool Zephyr::Graphics::CommandList::ready()
{
	return bClosed;
}

bool Zephyr::Graphics::CommandList::startRecording(const int fenceIndex)
{
	// can't start record on the same allocator before stoping the recording and after sending to GPU
	if (!mFences[fenceIndex]->isDone())
	{
		std::cout << "Attempt to record to command list & allocator that is still recording" << std::endl;
		return false;
	}
	mFences[fenceIndex]->increment();

	// we can only reset an allocator once the gpu is done with it
	// resetting an allocator frees the memory that the command list was stored in
	auto hr = mCommandAllocators[fenceIndex]->Reset();
	if (FAILED(hr))
	{
		return false;
	}

	// reset the command list. by resetting the command list we are putting it into
	// a recording state so we can start recording commands into the command allocator.
	// the command allocator that we reference here may have multiple command lists
	// associated with it, but only one can be recording at any time. Make sure
	// that any other command lists associated to this command allocator are in
	// the closed state (not recording).
	hr = mpCommandList->Reset(mCommandAllocators[fenceIndex], mpPipelineState->getPipeline());
	if (FAILED(hr))
	{
		return false;
	}
	bClosed = false;

	// here we start recording commands into the commandList (which all the commands will be stored in the commandAllocator)
	// can record after here, remember to end commandlist recording
	return true;
}

bool Zephyr::Graphics::CommandList::endRecording(const int frameIndex)
{
	auto hr = mpCommandList->Close();
	if (FAILED(hr))
	{
		return false;
	}
	bClosed = true;

	return true;
}

ID3D12GraphicsCommandList* Zephyr::Graphics::CommandList::getCommandList()
{
	return mpCommandList.get();
}

ID3D12PipelineState * Zephyr::Graphics::CommandList::getPipelineState() const
{
	return mpPipelineState->getPipeline();
}

bool Zephyr::Graphics::CommandList::createCommandAllocator(const int frameBufferCount)
{
	// -- Create the Command Allocators -- //
	mCommandAllocators.resize(frameBufferCount);
	for (int i = 0; i < frameBufferCount; ++i)
	{
		auto hr = mpDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCommandAllocators[i]));
		if (FAILED(hr))
		{
			return false;
		}
	}
	return true;
}

bool Zephyr::Graphics::CommandList::createCommandList()
{
	// create the command list with the first allocator
	ID3D12GraphicsCommandList* pCommandList = nullptr;
	auto hr = mpDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCommandAllocators[0], NULL, IID_PPV_ARGS(&pCommandList));
	if (FAILED(hr) || nullptr == pCommandList)
	{
		return false;
	}
	// command lists are created in the recording state. our main loop will set it up for recording again so close it now
	pCommandList->Close();
	bClosed = true;

	mpCommandList.reset(pCommandList);
	return true;
}

bool Zephyr::Graphics::CommandList::createFence(const int frameBufferCount)
{
	// -- Create a Fence & Fence Event -- //
	mFences.resize(FRAME_BUFFER_COUNT);
	for (int i = 0; i < FRAME_BUFFER_COUNT; ++i)
	{
		mFences[i] = std::shared_ptr<Fence>(new Fence(mpDevice));
	}

	return true;
}
