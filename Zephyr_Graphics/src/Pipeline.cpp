#include "Pipeline.h"
#include "Zephyr_Graphics.h"

Zephyr::Graphics::Pipeline::Pipeline(GraphicsEngine * pEngine) : mpEngine(pEngine), mpPipelineState(nullptr), mpRootSignature(nullptr)
{
}

Zephyr::Graphics::Pipeline::~Pipeline()
{
	SAFE_RELEASE(mpRootSignature);
	SAFE_RELEASE(mpPipelineState);
	SAFE_RELEASE(mpDSDescriptorHeap);
}

bool Zephyr::Graphics::Pipeline::initialize(const PipelineOption& option)
{
	auto success = createSampler();
	if (!success)
		return false;

	success = createRootSignature(option.rootSignatureFlag);
	if (!success)
		return false;

	success = createConstantBuffer(option.frameBufferCount, option.constantBufferSize);
	if (!success)
		return false;

	success = createInputLayout(option.inputLayout);
	if (!success)
		return false;

	success = createShaders(option.vertexShader, option.vertexShaderPath, option.pixelShader, option.pixelShaderPath);
	if (!success)
		return false;

	success = createDepthStencil();
	if (!success)
		return false;

	success = createPipeline();
	if (!success)
		return false;

	return true;
}

ID3D12PipelineState * Zephyr::Graphics::Pipeline::getPipeline() const
{
	return mpPipelineState;
}

ID3D12RootSignature * Zephyr::Graphics::Pipeline::getRootSignature() const
{
	return mpRootSignature;
}

ID3D12DescriptorHeap * Zephyr::Graphics::Pipeline::getDepthStencilDescriptorHeap() const
{
	return mpDSDescriptorHeap;
}

UINT8 * Zephyr::Graphics::Pipeline::getConstantBufferGPUAddress(const int frameIndex) const
{
	return mConstantBufferGPUAddress[frameIndex];
}

ID3D12Resource * Zephyr::Graphics::Pipeline::getConstantBufferUploadHeap(const int frameIndex) const
{
	return mConstantBufferUploadHeap[frameIndex];
}

ID3D12DescriptorHeap * Zephyr::Graphics::Pipeline::getConstantBufferDescriptorHeap(const int frameIndex) const
{
	return mConstantBufferDescriptorHeap[frameIndex];
}

bool Zephyr::Graphics::Pipeline::createRootSignature(const D3D12_ROOT_SIGNATURE_FLAGS& flag)
{
	auto renderer = mpEngine->getRenderer();

	// create a root descriptor, which explains where to find the data for this root parameter
	D3D12_ROOT_DESCRIPTOR rootCBVDescriptor;
	rootCBVDescriptor.RegisterSpace = 0;
	rootCBVDescriptor.ShaderRegister = 0;

	// create a descriptor range (descriptor table) and fill it out
	// this is a range of descriptors inside a descriptor heap
	D3D12_DESCRIPTOR_RANGE  descriptorTableRanges[1]; // only one range right now
	descriptorTableRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; // this is a range of constant buffer views (descriptors)
	descriptorTableRanges[0].NumDescriptors = 1; // we only have one constant buffer, so the range is only 1
	descriptorTableRanges[0].BaseShaderRegister = 0; // start index of the shader registers in the range
	descriptorTableRanges[0].RegisterSpace = 0; // space 0. can usually be zero
	descriptorTableRanges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND; // this appends the range to the end of the root signature descriptor tables

	// create a descriptor table
	D3D12_ROOT_DESCRIPTOR_TABLE descriptorTable;
	descriptorTable.NumDescriptorRanges = _countof(descriptorTableRanges); // we only have one range
	descriptorTable.pDescriptorRanges = &descriptorTableRanges[0]; // the pointer to the beginning of our ranges array

	// create a root parameter and fill it out
	D3D12_ROOT_PARAMETER  rootParameters[2]; // only one parameter right now
	rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; // this is a descriptor table
	rootParameters[0].Descriptor = rootCBVDescriptor; // this is our descriptor table for this root parameter
	rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX; // our pixel shader will be the only shader accessing this parameter for now

	// fill out the parameter for our descriptor table. Remember it's a good idea to sort parameters by frequency of change. Our constant
	// buffer will be changed multiple times per frame, while our descriptor table will not be changed at all (in this tutorial)
	rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; // this is a descriptor table
	rootParameters[1].DescriptorTable = descriptorTable; // this is our descriptor table for this root parameter
	rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL; // our pixel shader will be the only shader accessing this parameter for now

	CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
	rootSignatureDesc.Init(_countof(rootParameters), // we have 1 root parameter
		rootParameters, // a pointer to the beginning of our root parameters array
		1,
		&mSamplerDesc[0],
		D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | // we can deny shader stages here for better performance
		D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
		D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS);

	ID3DBlob* signature;
	auto hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
	if (FAILED(hr))
	{
		return false;
	}

	hr = renderer->getDevice()->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&mpRootSignature));
	if (FAILED(hr))
	{
		return false;
	}

	return true;
}

bool Zephyr::Graphics::Pipeline::createConstantBuffer(const int frameBufferCount, const int constantBufferSize)
{
	auto pDevice = mpEngine->getRenderer()->getDevice();

	mConstantBufferDescriptorHeap.resize(frameBufferCount);
	for (int i = 0; i < frameBufferCount; ++i)
	{
		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
		heapDesc.NumDescriptors = 1;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		auto hr = pDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mConstantBufferDescriptorHeap[i]));
		if (FAILED(hr))
		{
			return false;
		}
	}

	// create the constant buffer resource heap
	// We will update the constant buffer one or more times per frame, so we will use only an upload heap
	// unlike previously we used an upload heap to upload the vertex and index data, and then copied over
	// to a default heap. If you plan to use a resource for more than a couple frames, it is usually more
	// efficient to copy to a default heap where it stays on the gpu. In this case, our constant buffer
	// will be modified and uploaded at least once per frame, so we only use an upload heap

	// create a resource heap, descriptor heap, and pointer to cbv for each frame
	mConstantBufferUploadHeap.resize(frameBufferCount);
	mConstantBufferGPUAddress.resize(frameBufferCount);
	for (int i = 0; i < frameBufferCount; ++i)
	{
		auto hr = pDevice->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), // this heap will be used to upload the constant buffer data
			D3D12_HEAP_FLAG_NONE, // no flags
			&CD3DX12_RESOURCE_DESC::Buffer(1024 * 64), // size of the resource heap. Must be a multiple of 64KB for single-textures and constant buffers
			D3D12_RESOURCE_STATE_GENERIC_READ, // will be data that is read from so we keep it in the generic read state
			nullptr, // we do not have use an optimized clear value for constant buffers
			IID_PPV_ARGS(&mConstantBufferUploadHeap[i]));
		if (FAILED(hr))
			return false;

		mConstantBufferUploadHeap[i]->SetName(L"Constant Buffer Upload Resource Heap");

		D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
		cbvDesc.BufferLocation = mConstantBufferUploadHeap[i]->GetGPUVirtualAddress();
		cbvDesc.SizeInBytes = (constantBufferSize + 255) & ~255;    // CB size is required to be 256-byte aligned.
		pDevice->CreateConstantBufferView(&cbvDesc, mConstantBufferDescriptorHeap[i]->GetCPUDescriptorHandleForHeapStart());

		CD3DX12_RANGE readRange(0, 0);    // We do not intend to read from this resource on the CPU. (End is less than or equal to begin)
		hr = mConstantBufferUploadHeap[i]->Map(0, &readRange, reinterpret_cast<void**>(&mConstantBufferGPUAddress[i]));
		//memcpy(mConstantBufferGPUAddress[i], &cbColorMultiplierData, sizeof(cbColorMultiplierData));
	}

	return true;
}

bool Zephyr::Graphics::Pipeline::createInputLayout(const std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout)
{
	if (inputLayout.empty())
		return false;

	mInputElements = inputLayout;

	// fill out an input layout description structure
	// we can get the number of elements in an array by "sizeof(array) / sizeof(arrayElementType)"
	mInputLayout.NumElements = (UINT)mInputElements.size();
	mInputLayout.pInputElementDescs = &mInputElements[0];

	return true;
}

bool Zephyr::Graphics::Pipeline::createShaders(const std::wstring & vertexShader, const std::wstring & vertexShaderPath, const std::wstring & pixelShader, const std::wstring & pixelShaderPath)
{
	auto shaderManager = mpEngine->getShaderManager();

	// first is vertex shader
	mShaderByteCodes.push_back(shaderManager->getOrCreateShader(vertexShaderPath, vertexShader, ShaderManager::VERTEX));
	// second is pixel shader
	mShaderByteCodes.push_back(shaderManager->getOrCreateShader(pixelShaderPath, pixelShader, ShaderManager::PIXEL));

	return true;
}

bool Zephyr::Graphics::Pipeline::createDepthStencil()
{
	// create a depth stencil descriptor heap so we can get a pointer to the depth stencil buffer
	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	auto hr = mpEngine->getRenderer()->getDevice()->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&mpDSDescriptorHeap));
	if (FAILED(hr))
	{
		return false;
	}

	D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
	depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

	D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
	depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
	depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
	depthOptimizedClearValue.DepthStencil.Stencil = 0;

	auto depthStencilResource = mpEngine->getResourceManager()->createDepthStencilResource(L"Depth/Stencil Resource Heap", depthOptimizedClearValue);
	if (depthStencilResource.isNull())
		return false;

	mpEngine->getRenderer()->getDevice()->CreateDepthStencilView(depthStencilResource.get(), &depthStencilDesc, mpDSDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

	return true;
}

bool Zephyr::Graphics::Pipeline::createSampler()
{
	// create a static sampler
	D3D12_STATIC_SAMPLER_DESC sampler;
	sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
	sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
	sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
	sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
	sampler.MipLODBias = 0;
	sampler.MaxAnisotropy = 0;
	sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
	sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
	sampler.MinLOD = 0.0f;
	sampler.MaxLOD = D3D12_FLOAT32_MAX;
	sampler.ShaderRegister = 0;
	sampler.RegisterSpace = 0;
	sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
	mSamplerDesc.push_back(sampler);

	return true;
}

bool Zephyr::Graphics::Pipeline::createPipeline()
{
	auto renderer = mpEngine->getRenderer();

	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {}; // a structure to define a pso
	psoDesc.InputLayout = mInputLayout; // the structure describing our input layout
	psoDesc.pRootSignature = mpRootSignature; // the root signature that describes the input data this pso needs
	psoDesc.VS = mShaderByteCodes[0]; // structure describing where to find the vertex shader bytecode and how large it is
	psoDesc.PS = mShaderByteCodes[1]; // same as VS but for pixel shader
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; // type of topology we are drawing
	psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM; // format of the render target
	psoDesc.SampleDesc = renderer->getSampleDesc(); // must be the same sample description as the swapchain and depth/stencil buffer
	psoDesc.SampleMask = 0xffffffff; // sample mask has to do with multi-sampling. 0xffffffff means point sampling is done
	psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT); // a default rasterizer state.
	psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT); // a default blent state.
	psoDesc.NumRenderTargets = 1; // we are only binding one render target
	psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;

	// create the pso
	auto hr = renderer->getDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mpPipelineState));
	if (FAILED(hr))
	{
		return false;
	}

	return true;
}
