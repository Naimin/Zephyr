#include "ResourceManager.h"
#include "Zephyr_Graphics.h"
#include <sstream>

Zephyr::Graphics::ResourceManager::ResourceManager(GraphicsEngine * pGraphicsEngine) : mpGraphicsEngine(pGraphicsEngine), mFence(pGraphicsEngine->getRenderer()->getDevice())
{
}

Zephyr::Graphics::ResourceManager::~ResourceManager()
{
	mpCommandQueue->wait();
}

bool Zephyr::Graphics::ResourceManager::initialize()
{
	if (!createCommandQueue())
		return false;

	return true;
}

Zephyr::SharedPtr<ID3D12Resource> Zephyr::Graphics::ResourceManager::createResource(const std::wstring & resourceName, int bufferSize, RESOURCE_TYPE type, D3D12_RESOURCE_DESC* description)
{
	auto itr = mResources.find(resourceName);
	if (itr != mResources.end()) // if already created just return
		return itr->second;

	auto device = mpGraphicsEngine->getRenderer()->getDevice();

	D3D12_RESOURCE_STATES resourceState;
	switch (type)
	{
		case DEFAULT: resourceState = D3D12_RESOURCE_STATE_COPY_DEST; break;
		case UPLOAD: resourceState = D3D12_RESOURCE_STATE_GENERIC_READ; break;
	}

	ID3D12Resource* pBuffer = nullptr;
	auto hr = device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE(type)), // a default heap
		D3D12_HEAP_FLAG_NONE, // no flags
		description ? description : &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), // resource description for a buffer
		resourceState, // we will start this heap in the copy destination state since we will copy data
										// from the upload heap to this heap
		nullptr, // optimized clear value must be null for this type of resource. used for render targets and depth/stencil buffers
		IID_PPV_ARGS(&pBuffer));
	if (FAILED(hr))
		return nullptr;

	pBuffer->SetName(resourceName.c_str());

	auto pResource = SharedPtr<ID3D12Resource>(pBuffer);
	mResources[resourceName] = pResource;
	mResourceSize[resourceName] = bufferSize;

	return pResource;
}

Zephyr::SharedPtr<ID3D12Resource> Zephyr::Graphics::ResourceManager::createDepthStencilResource(const std::wstring & resourceName, D3D12_CLEAR_VALUE& clearValue)
{
	auto itr = mResources.find(resourceName);
	if (itr != mResources.end()) // if already created just return
		return itr->second;

	auto device = mpGraphicsEngine->getRenderer()->getDevice();

	ID3D12Resource* pBuffer = nullptr;
	auto hr = device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // a default heap
		D3D12_HEAP_FLAG_NONE, // no flags
		&CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, mpGraphicsEngine->getWidth(), mpGraphicsEngine->getHeight(), 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
		D3D12_RESOURCE_STATE_DEPTH_WRITE, // we will start this heap in the copy destination state since we will copy data
					   // from the upload heap to this heap
		&clearValue, // optimized clear value must be null for this type of resource. used for render targets and depth/stencil buffers
		IID_PPV_ARGS(&pBuffer));
	if (FAILED(hr))
		return nullptr;

	pBuffer->SetName(resourceName.c_str());

	auto pResource = SharedPtr<ID3D12Resource>(pBuffer);
	mResources[resourceName] = pResource;
	mResourceSize[resourceName] = mpGraphicsEngine->getWidth() * mpGraphicsEngine->getHeight() * sizeof(float);

	return pResource;
}

bool Zephyr::Graphics::ResourceManager::copyDataToResource(const std::wstring & toResourceName, const std::wstring & fromResourceName, int bufferSize, void * pData, int rowPitch, int height)
{
	if (toResourceName.empty() || fromResourceName.empty() || bufferSize <= 0 || nullptr == pData)
		return false;
	
	auto toItr = mResources.find(toResourceName);
	auto fromItr = mResources.find(fromResourceName);
	if (toItr == mResources.end() || fromItr == mResources.end())
		return false;

	// store vertex buffer in upload heap
	D3D12_SUBRESOURCE_DATA pTransferData = {};
	pTransferData.pData = reinterpret_cast<BYTE*>(pData); // pointer to our vertex array
	pTransferData.RowPitch = rowPitch == -1 ? bufferSize : rowPitch; // size of all our triangle vertex data
	pTransferData.SlicePitch = rowPitch == -1 ? bufferSize : (rowPitch * height); // also the size of our triangle vertex data

	std::shared_ptr<CommandList> pCommand(new CommandList(1, mpGraphicsEngine));
	pCommand->startRecording(0);
	// we are now creating a command with the command list to copy the data from
	// the upload heap to the default heap
	UpdateSubresources(pCommand->getCommandList(), toItr->second.get(), fromItr->second.get(), 0, 0, 1, &pTransferData);

	// transition the vertex buffer data from copy destination state to vertex buffer state
	pCommand->getCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(toItr->second.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));
	pCommand->endRecording(0);

	// wait for previous 
	mpCommandQueue->wait();

	mpCommandQueue->enqueueCommandList(pCommand);

	mpCommandQueue->execute(0);

	mpCommandQueue->clear();

	return true;
}

bool Zephyr::Graphics::ResourceManager::createAndCopyToGPU(const std::wstring & resourceName, int bufferSize, void * pData, int rowPitch, int height)
{
	// resource already created
	if (mResources.find(resourceName) != mResources.end())
		return true;

	std::wstringstream uploadHeapName;
	uploadHeapName << resourceName << L"_UH";
	auto uploadHeap = createResource(uploadHeapName.str(), bufferSize, ResourceManager::UPLOAD);
	if (uploadHeap.isNull())
		return false;

	std::wstring defaultHeapName = resourceName;
	auto defaultHeap = createResource(resourceName, bufferSize, ResourceManager::DEFAULT);
	if (defaultHeap.isNull())
		return false;

	auto success = copyDataToResource(resourceName, uploadHeapName.str(), bufferSize, pData, rowPitch, height);
	if (!success)
		return false;

	// wait for the transfer to finish
	waitForPreviousTask();

	// release the upload heap, since it is not use anymore
	releaseResource(uploadHeapName.str());

	return true;
}

bool Zephyr::Graphics::ResourceManager::createTextureAndCopyToGPU(const std::wstring & resourceName, D3D12_RESOURCE_DESC* description, int bufferSize, void * pData, int rowPitch, int height)
{
	// resource already created
	if (mResources.find(resourceName) != mResources.end())
		return true;

	std::wstringstream uploadHeapName;
	uploadHeapName << resourceName << L"_UH";
	auto uploadHeap = createResource(uploadHeapName.str(), bufferSize, ResourceManager::UPLOAD);
	if (uploadHeap.isNull())
		return false;

	std::wstring defaultHeapName = resourceName;
	auto defaultHeap = createResource(resourceName, bufferSize, ResourceManager::DEFAULT, description);
	if (defaultHeap.isNull())
		return false;

	auto success = copyTextureToResource(resourceName, uploadHeapName.str(), bufferSize, pData, rowPitch, height);
	if (!success)
		return false;

	// wait for the transfer to finish
	waitForPreviousTask();

	// release the upload heap, since it is not use anymore
	releaseResource(uploadHeapName.str());

	return true;
}

bool Zephyr::Graphics::ResourceManager::copyTextureToResource(const std::wstring & toResourceName, const std::wstring & fromResourceName, int bufferSize, void * pData, int rowPitch, int height)
{
	if (toResourceName.empty() || fromResourceName.empty() || bufferSize <= 0 || nullptr == pData)
		return false;

	auto toItr = mResources.find(toResourceName);
	auto fromItr = mResources.find(fromResourceName);
	if (toItr == mResources.end() || fromItr == mResources.end())
		return false;

	// store vertex buffer in upload heap
	D3D12_SUBRESOURCE_DATA pTransferData = {};
	pTransferData.pData = reinterpret_cast<BYTE*>(pData); // pointer to our vertex array
	pTransferData.RowPitch = rowPitch == -1 ? bufferSize : rowPitch; // size of all our triangle vertex data
	pTransferData.SlicePitch = rowPitch == -1 ? bufferSize : (rowPitch * height); // also the size of our triangle vertex data

	std::shared_ptr<CommandList> pCommand(new CommandList(1, mpGraphicsEngine));
	pCommand->startRecording(0);
	// we are now creating a command with the command list to copy the data from
	// the upload heap to the default heap
	UpdateSubresources(pCommand->getCommandList(), toItr->second.get(), fromItr->second.get(), 0, 0, 1, &pTransferData);

	// transition the texture default heap to a pixel shader resource (we will be sampling from this heap in the pixel shader to get the color of pixels)
	pCommand->getCommandList()->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(toItr->second.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
	pCommand->endRecording(0);

	// wait for previous 
	mpCommandQueue->wait();

	mpCommandQueue->enqueueCommandList(pCommand);

	mpCommandQueue->execute(0);

	mpCommandQueue->clear();

	return true;
}

Zephyr::SharedPtr<ID3D12Resource> Zephyr::Graphics::ResourceManager::getResource(const std::wstring & resourceName)
{
	auto itr = mResources.find(resourceName);
	if (itr == mResources.end())
		return nullptr;

	return itr->second;
}

void Zephyr::Graphics::ResourceManager::releaseResource(const std::wstring & resourceName)
{
	waitForPreviousTask();
	mResources.erase(resourceName);
	mResourceSize.erase(resourceName);
}

void Zephyr::Graphics::ResourceManager::waitForPreviousTask()
{
	mpCommandQueue->wait();
}

D3D12_VERTEX_BUFFER_VIEW Zephyr::Graphics::ResourceManager::getVertexResourceView(const std::wstring & resourceName, const int stride)
{
	auto itr = mResources.find(resourceName);
	if (itr == mResources.end())
		return D3D12_VERTEX_BUFFER_VIEW();

	D3D12_VERTEX_BUFFER_VIEW bufferView;

	// create a vertex buffer view for the triangle. We get the GPU memory address to the vertex pointer using the GetGPUVirtualAddress() method
	bufferView.BufferLocation = itr->second->GetGPUVirtualAddress();
	bufferView.StrideInBytes = stride;
	bufferView.SizeInBytes = mResourceSize[resourceName];

	return bufferView;
}

D3D12_INDEX_BUFFER_VIEW Zephyr::Graphics::ResourceManager::getIndexResourceView(const std::wstring & resourceName)
{
	auto itr = mResources.find(resourceName);
	if (itr == mResources.end())
		return D3D12_INDEX_BUFFER_VIEW();

	D3D12_INDEX_BUFFER_VIEW bufferView;
	// create a index buffer view for the triangle. We get the GPU memory address to the index pointer using the GetGPUVirtualAddress() method
	bufferView.BufferLocation = itr->second->GetGPUVirtualAddress();
	bufferView.SizeInBytes = mResourceSize[resourceName];
	bufferView.Format = DXGI_FORMAT_R32_UINT;

	return bufferView;
}

Zephyr::Graphics::GraphicsEngine * Zephyr::Graphics::ResourceManager::getEngine()
{
	return mpGraphicsEngine;
}

bool Zephyr::Graphics::ResourceManager::createCommandQueue()
{
	auto device = mpGraphicsEngine->getRenderer()->getDevice();

	mpCommandQueue.reset(new CommandQueue(0, COMMAND_QUEUE_TYPE::DIRECT, device));

	return true;
}
