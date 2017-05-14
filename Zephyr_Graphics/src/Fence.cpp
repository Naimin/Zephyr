#include "Fence.h"

Zephyr::Graphics::Fence::Fence(SharedPtr<ID3D12Device> pDevice)
{
	auto hr = pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence));
	if (FAILED(hr))
	{
		return;
	}
	mFenceValue = 0; // set the initial fence value to 0

	// create a handle to a fence event
	mFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (nullptr == mFenceEvent)
	{
		return;
	}
}

Zephyr::Graphics::Fence::~Fence()
{
	SAFE_RELEASE(mFence);
	CloseHandle(mFenceEvent);
}

UINT64 Zephyr::Graphics::Fence::current()
{
	return mFence->GetCompletedValue();
}

void Zephyr::Graphics::Fence::increment()
{
	++mFenceValue;
}

void Zephyr::Graphics::Fence::signal()
{
	mFence->Signal(mFence->GetCompletedValue() + 1);
}

bool Zephyr::Graphics::Fence::signal(SharedPtr<ID3D12CommandQueue> pCommandQueue)
{
	auto hr = pCommandQueue->Signal(mFence, mFenceValue);
	return SUCCEEDED(hr);
}

bool Zephyr::Graphics::Fence::isDone()
{
	UINT64 value = mFence->GetCompletedValue();
	return value >= mFenceValue;
}

void Zephyr::Graphics::Fence::waitForFence()
{
	if (!isDone()) // if not done then wait
	{
		// we have the fence create an event which is signaled once the fence's current value is "fenceValue"
		auto hr = mFence->SetEventOnCompletion(mFenceValue, mFenceEvent);
		if (FAILED(hr))
		{
			return;
		}

		// We will wait until the fence has triggered the event that it's current value has reached "fenceValue". once it's value
		// has reached "fenceValue", we know the command queue has finished executing
		WaitForSingleObject(mFenceEvent, INFINITE);
	}
}

void Zephyr::Graphics::Fence::waitAndIncrement()
{
	waitForFence();
	increment();
}
