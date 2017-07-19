#include "Renderer.h"
#include <tbb/parallel_for.h>

using namespace Zephyr;

Zephyr::Graphics::Renderer::Renderer(GraphicsEngine* pEngine) : mpEngine(pEngine)
{
}

Zephyr::Graphics::Renderer::~Renderer()
{
	cleanup();
}

bool Zephyr::Graphics::Renderer::initialize(unsigned int backBufferWidth, unsigned int backBufferHeight, HWND& hwnd, bool bFullScreen)
{
	bool success = createDevice();
	if (!success)
		return false;

	success = createCommandQueue();
	if (!success)
		return false;

	success = createSwapChain(hwnd, bFullScreen);
	if (!success)
		return false;

	success = createRenderTargetView();
	if (!success)
		return false;

	success = createFence();
	if (!success)
		return false;

	mpDxgiFactory.reset();
	bIsRunning = true;

	return true;
}

void Zephyr::Graphics::Renderer::addRenderPass(const std::string& RenderPassName, IRenderPass * pRenderPass)
{
	if (mRenderPassMap.find(RenderPassName) != mRenderPassMap.end())
		std::cout << "Warning: Render pass name already exist and will override the old one." << std::endl;
	mRenderPassMap[RenderPassName].reset(pRenderPass);
}

bool Zephyr::Graphics::Renderer::enqueuRenderPass(const std::string& RenderPassName, const int queueIndex)
{
	if (mRenderPassMap.find(RenderPassName) == mRenderPassMap.end())
		return false;

	// keep creating a new commandQueue till the queue index is reached
	while (queueIndex >= mCommandQueues.size())
	{
		createCommandQueue();
	}

	// enqueue the render pass into the command queue
	mCommandQueues[queueIndex]->enqueueCommandList(mRenderPassMap[RenderPassName]);
	
	return true;
}

void Zephyr::Graphics::Renderer::clearRenderPassQueue(const int queueIndex)
{
	// out of range
	if (queueIndex >= mCommandQueues.size())
		return;

	mCommandQueues[queueIndex]->clear();
}

bool Zephyr::Graphics::Renderer::enqueuUIRenderPass(const std::string & renderPassName, const int queueIndex)
{
	if (mRenderPassMap.find(renderPassName) == mRenderPassMap.end())
		return false;

	// keep creating a new commandQueue till the queue index is reached
	while (queueIndex >= mUICommandQueues.size())
	{
		createUICommandQueue();
	}

	// enqueue the render pass into the command queue
	mUICommandQueues[queueIndex]->enqueueCommandList(mRenderPassMap[renderPassName]);

	return true;
}

void Zephyr::Graphics::Renderer::clearUIRenderPassQueue(const int queueIndex)
{
	// out of range
	if (queueIndex >= mUICommandQueues.size())
		return;

	mUICommandQueues[queueIndex]->clear();
}

void Zephyr::Graphics::Renderer::render()
{
	HRESULT hr = getDevice()->GetDeviceRemovedReason();
	if (S_OK != hr)
		std::cout << hr;

	// We have to wait for the gpu to finish with the command allocator before we reset it
	waitForPreviousFrame();

	// Get the delta time since the previous render is called
	double deltaTime = mTimer.getDeltaTime();

	mFences[mFrameIndex]->increment();

	// composite both the normal render pass with the UI render pass
	std::vector<std::shared_ptr<CommandQueue>> commandQueues;

	for (auto commandQueue : mCommandQueues)
	{
		commandQueues.push_back(commandQueue);
	}
	for (auto commandQueue : mUICommandQueues)
	{
		commandQueues.push_back(commandQueue);
	}

	int i = 0;
	for (auto commandQueue : commandQueues)
	{
		commandQueue->update(mFrameIndex, deltaTime);
		//update(i++);
	}

	// execute the array of command lists
	for (auto commandQueue : commandQueues)
	{
		commandQueue->execute(mFrameIndex);
		
		hr = getDevice()->GetDeviceRemovedReason();
		if (S_OK != hr)
			std::cout << hr;

		commandQueue->wait();
	}

	// signal that the render is done
	mFences[mFrameIndex]->signal();

	// present the current backbuffer
	hr = mpSwapChain->Present(0, 0);
	if (FAILED(hr))
	{
		hr = mpDevice->GetDeviceRemovedReason();
		std::cout << hr << std::endl;
	}
}

void Zephyr::Graphics::Renderer::cleanup()
{
	// we want to wait for the gpu to finish executing the command list before we start releasing everything
	waitForPreviousFrame();

	// wait for the gpu to finish all frames
	for (int i = 0; i < FRAME_BUFFER_COUNT; ++i)
	{
		mFrameIndex = i;
		waitForPreviousFrame();
	}

	// get swapchain out of full screen before exiting
	BOOL fs = false;
	if (mpSwapChain->GetFullscreenState(&fs, NULL))
		mpSwapChain->SetFullscreenState(false, NULL);
}

void Zephyr::Graphics::Renderer::waitForPreviousFrame()
{
	// swap the current rtv buffer index so we draw on the correct buffer
	mFrameIndex = mpSwapChain->GetCurrentBackBufferIndex();

	// increment fenceValue for next frame
	mFences[mFrameIndex]->waitForFence();
}

bool Zephyr::Graphics::Renderer::isRunning() const
{
	return bIsRunning;
}

SharedPtr<ID3D12Device> Zephyr::Graphics::Renderer::getDevice() const
{
	return mpDevice;
}

Zephyr::SharedPtr<ID3D12DescriptorHeap> Zephyr::Graphics::Renderer::getDescriptionHeap() const
{
	return mpRtvDescriptorHeap;
}

int Zephyr::Graphics::Renderer::getFrameIndex() const
{
	return mFrameIndex;
}

int Zephyr::Graphics::Renderer::getRtvDescriptorSize() const
{
	return mRtvDescriptorSize;
}

std::vector<ID3D12Resource*> Zephyr::Graphics::Renderer::getRenderTargets() const
{
	return mRenderTargets;
}

DXGI_SAMPLE_DESC Zephyr::Graphics::Renderer::getSampleDesc() const
{
	return mSampleDesc;
}

bool Zephyr::Graphics::Renderer::createDevice()
{
	// -- Create the Device -- //
	IDXGIFactory4* pDxgiFactory;
	auto hr = CreateDXGIFactory1(IID_PPV_ARGS(&pDxgiFactory));
	if (FAILED(hr))
	{
		return false;
	}
	mpDxgiFactory.reset(pDxgiFactory);

	IDXGIAdapter1* adapter; // adapters are the graphics card (this includes the embedded graphics on the motherboard)

	int adapterIndex = 0; // we'll start looking for directx 12  compatible graphics devices starting at index 0

	bool adapterFound = false; // set this to true when a good one was found

							   // find first hardware gpu that supports d3d 12
	while (mpDxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND)
	{
		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1(&desc);

		if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
		{
			// we dont want a software device
			adapterIndex++; // add this line here. Its not currently in the downloadable project
			continue;
		}

		// we want a device that is compatible with direct3d 12 (feature level 11 or higher)
		hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr);
		if (SUCCEEDED(hr))
		{
			adapterFound = true;
			break;
		}

		++adapterIndex;
	}

	if (!adapterFound)
	{
		return false;
	}

	ID3D12Device* pDevice = nullptr;
	// Create the device
	hr = D3D12CreateDevice(
		adapter,
		D3D_FEATURE_LEVEL_12_1,
		IID_PPV_ARGS(&pDevice)
		);

	if (FAILED(hr) || nullptr == pDevice)
	{
		return false;
	}
	mpDevice.reset(pDevice);

	return true;
}

bool Zephyr::Graphics::Renderer::createCommandQueue()
{
	if (mpDevice.isNull())
		return false;

	mCommandQueues.push_back(std::shared_ptr<CommandQueue>(new CommandQueue((int)mCommandQueues.size(), COMMAND_QUEUE_TYPE::DIRECT, mpDevice)));
	return true;
}

bool Zephyr::Graphics::Renderer::createUICommandQueue()
{
	if (mpDevice.isNull())
		return false;

	mUICommandQueues.push_back(std::shared_ptr<CommandQueue>(new CommandQueue((int)mCommandQueues.size(), COMMAND_QUEUE_TYPE::DIRECT, mpDevice)));
	return true;
}

bool Zephyr::Graphics::Renderer::createSwapChain(HWND& hwnd, bool bFullscreen)
{
	// -- Create the Swap Chain (double/tripple buffering) -- //
	DXGI_MODE_DESC backBufferDesc = {}; // this is to describe our display mode
	backBufferDesc.Width = mBackBufferWidth; // buffer width
	backBufferDesc.Height = mBackBufferHeight; // buffer height
	backBufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // format of the buffer (rgba 32 bits, 8 bits for each chanel)

	// describe our multi-sampling. We are not multi-sampling, so we set the count to 1 (we need at least one sample of course)
	mSampleDesc.Count = 1; // multisample count (no multisampling, so we just put 1, since we still need 1 sample)

						  // Describe and create the swap chain.
	DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
	swapChainDesc.BufferCount = FRAME_BUFFER_COUNT; // number of buffers we have
	swapChainDesc.BufferDesc = backBufferDesc; // our back buffer description
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; // this says the pipeline will render to this swap chain
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; // dxgi will discard the buffer (data) after we call present
	swapChainDesc.OutputWindow = hwnd; // handle to our window
	swapChainDesc.SampleDesc = mSampleDesc; // our multi-sampling description
	swapChainDesc.Windowed = !bFullscreen; // set to true, then if in fullscreen must call SetFullScreenState with true for full screen to get uncapped fps

	IDXGISwapChain* tempSwapChain;

	mpDxgiFactory->CreateSwapChain(
		mCommandQueues[0]->getCommandQueue().get(), // the queue will be flushed once the swap chain is created
		&swapChainDesc, // give it the swap chain description we created above
		&tempSwapChain // store the created swap chain in a temp IDXGISwapChain interface
		);

	if (nullptr == tempSwapChain)
		return false;

	auto pSwapChain = static_cast<IDXGISwapChain3*>(tempSwapChain);
	mpSwapChain.reset(pSwapChain);

	mFrameIndex = mpSwapChain->GetCurrentBackBufferIndex();

	return true;
}

bool Zephyr::Graphics::Renderer::createRenderTargetView()
{
	// -- Create the Back Buffers (render target views) Descriptor Heap -- //

	// describe an rtv descriptor heap and create
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
	rtvHeapDesc.NumDescriptors = FRAME_BUFFER_COUNT; // number of descriptors for this heap.
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; // this heap is a render target view heap

	// This heap will not be directly referenced by the shaders (not shader visible), as this will store the output from the pipeline
	// otherwise we would set the heap's flag to D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	ID3D12DescriptorHeap* pRtvDescriptorHeap = nullptr;
	auto hr = mpDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&pRtvDescriptorHeap));
	if (FAILED(hr))
	{
		return false;
	}
	mpRtvDescriptorHeap.reset(pRtvDescriptorHeap);

	// get the size of a descriptor in this heap (this is a rtv heap, so only rtv descriptors should be stored in it.
	// descriptor sizes may vary from device to device, which is why there is no set size and we must ask the 
	// device to give us the size. we will use this size to increment a descriptor handle offset
	mRtvDescriptorSize = mpDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	// get a handle to the first descriptor in the descriptor heap. a handle is basically a pointer,
	// but we cannot literally use it like a c++ pointer.
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(mpRtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	// Create a RTV for each buffer (double buffering is two buffers, tripple buffering is 3).
	mRenderTargets.resize(FRAME_BUFFER_COUNT);
	for (int i = 0; i < FRAME_BUFFER_COUNT; ++i)
	{
		// first we get the n'th buffer in the swap chain and store it in the n'th
		// position of our ID3D12Resource array
		hr = mpSwapChain->GetBuffer(i, IID_PPV_ARGS(&mRenderTargets[i]));
		if (FAILED(hr))
		{
			return false;
		}

		// the we "create" a render target view which binds the swap chain buffer (ID3D12Resource[n]) to the rtv handle
		mpDevice->CreateRenderTargetView(mRenderTargets[i], nullptr, rtvHandle);

		// we increment the rtv handle by the rtv descriptor size we got above
		rtvHandle.Offset(1, mRtvDescriptorSize);
	}

	return true;
}

bool Zephyr::Graphics::Renderer::createFence()
{
	// -- Create a Fence & Fence Event -- //
	// create the fences
	mFences.resize(FRAME_BUFFER_COUNT);
	for (int i = 0; i < FRAME_BUFFER_COUNT; ++i)
	{
		mFences[i] = std::shared_ptr<Fence>(new Fence(mpDevice));
	}

	return true;
}
