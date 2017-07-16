#include "BasicRenderPass.h"
#include "Renderer.h"
#include "Zephyr_Graphics.h"
#include "Vertex.h"
#include "Pipeline.h"
#include "RenderableModel.h"
#include <boost/algorithm/string.hpp>

using namespace DirectX;

Zephyr::Graphics::BasicRenderPass::BasicRenderPass(const int frameBufferCount, GraphicsEngine* pEngine) : IRenderPass(frameBufferCount, pEngine)
{
	setClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

Zephyr::Graphics::BasicRenderPass::~BasicRenderPass()
{
}

void Zephyr::Graphics::BasicRenderPass::setClearColor(float r, float g, float b, float a)
{
	clearColor[0] = r;
	clearColor[1] = b;
	clearColor[2] = g;
	clearColor[3] = a;
}

bool Zephyr::Graphics::BasicRenderPass::loadModel(const std::string & modelPath)
{
	mpEngine->waitForPreviousFrame();
	
	auto resourceManager = mpEngine->getResourceManager();
	// wait for the transfer to complete first 
	resourceManager->waitForPreviousTask();

	boost::filesystem::path filePath(modelPath);
	if (!boost::iequals(mModelPath, modelPath))
	{
		mModelPath = filePath.string();
		
		// discard previous model and create a new one
		mpModel.reset(new RenderableModel(filePath.wstring(), mpEngine->getResourceManager().get()));

		// load the new model
		auto success = mpModel->loadFromFile(mModelPath);
		if (!success)
			return false;

		success = mpModel->uploadToGPU();
		if (!success)
			return false;
	}

	return true;
}

bool Zephyr::Graphics::BasicRenderPass::initialize()
{
	std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayout;
	inputLayout.push_back({ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 });
	inputLayout.push_back({ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 });
	inputLayout.push_back({ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 });
	inputLayout.push_back({ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 });

	PipelineOption option;
	option.rootSignatureFlag = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
	option.inputLayout = inputLayout;
	option.vertexShader = L"vertexShader";
	option.vertexShaderPath = L"..\\shader\\vertex_shader.hlsl";
	option.pixelShader = L"pixelShader";
	option.pixelShaderPath = L"..\\shader\\pixel_shader.hlsl";
	option.frameBufferCount = 3;
	option.constantBufferSize = sizeof(ConstantBuffer);

	mpPipelineState->initialize(option);
	 
	// build projection and view matrix
	XMMATRIX tmpMat = XMMatrixPerspectiveFovLH(45.0f*(3.14f / 180.0f), (float)mpEngine->getWidth() / (float)mpEngine->getHeight(), 0.1f, 1000.0f);
	XMStoreFloat4x4(&cameraProjMat, tmpMat);

	// set starting camera state
	cameraPosition = XMFLOAT4(0.0f, 2.0f, -200.0f, 0.0f);
	cameraTarget = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
	cameraUp = XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);

	// build view matrix
	XMVECTOR cPos = XMLoadFloat4(&cameraPosition);
	XMVECTOR cTarg = XMLoadFloat4(&cameraTarget);
	XMVECTOR cUp = XMLoadFloat4(&cameraUp);
	tmpMat = XMMatrixLookAtLH(cPos, cTarg, cUp);
	XMStoreFloat4x4(&cameraViewMat, tmpMat);

	// set starting cubes position
	// first cube
	cube1Position = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f); // set cube 1's position
	XMVECTOR posVec = XMLoadFloat4(&cube1Position); // create xmvector for cube1's position

	tmpMat = XMMatrixTranslationFromVector(posVec); // create translation matrix from cube1's position vector
	XMStoreFloat4x4(&cube1RotMat, XMMatrixIdentity()); // initialize cube1's rotation matrix to identity matrix
	XMStoreFloat4x4(&cube1WorldMat, tmpMat); // store cube1's world matrix

	return true;
}

void Zephyr::Graphics::BasicRenderPass::update(const int frameIndex)
{
		// update app logic, such as moving the camera or figuring out what objects are in view

		// create rotation matrices
		XMMATRIX rotXMat = XMMatrixRotationX(0.0000f);
		XMMATRIX rotYMat = XMMatrixRotationY(0.0002f);
		XMMATRIX rotZMat = XMMatrixRotationZ(0.0000f);

		// add rotation to cube1's rotation matrix and store it
		XMMATRIX rotMat = XMLoadFloat4x4(&cube1RotMat) * rotXMat * rotYMat * rotZMat;
		XMStoreFloat4x4(&cube1RotMat, rotMat);

		// create translation matrix for cube 1 from cube 1's position vector
		XMMATRIX translationMat = XMMatrixTranslationFromVector(XMLoadFloat4(&cube1Position));

		// create cube1's world matrix by first rotating the cube, then positioning the rotated cube
		XMMATRIX worldMat = rotMat * translationMat;

		// store cube1's world matrix
		XMStoreFloat4x4(&cube1WorldMat, worldMat);

		// update constant buffer for cube1
		// create the wvp matrix and store in constant buffer
		XMMATRIX viewMat = XMLoadFloat4x4(&cameraViewMat); // load view matrix
		XMMATRIX projMat = XMLoadFloat4x4(&cameraProjMat); // load projection matrix
		XMMATRIX wvpMat = XMLoadFloat4x4(&cube1WorldMat) * viewMat * projMat; // create wvp matrix
		XMMATRIX transposed = XMMatrixTranspose(wvpMat); // must transpose wvp matrix for the gpu
		XMStoreFloat4x4(&mConstantBuffer.wvpMat, transposed); // store transposed wvp matrix in constant buffer

	// copy over the constant buffer value to gpu
	memcpy(mpPipelineState->getConstantBufferGPUAddress(frameIndex), &mConstantBuffer, sizeof(mConstantBuffer));

	startRecording(frameIndex);

	auto resourceManager = mpEngine->getResourceManager();
	// wait for the transfer to complete first 
	resourceManager->waitForPreviousTask();

	// Here go the actual rendering commands
	auto renderer = mpEngine->getRenderer();
	auto& commandList = mpCommandList;
	auto& renderTarget = renderer->getRenderTargets();

	// transition the "frameIndex" render target from the present state to the render target state so the command list draws to it starting from here
	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTarget[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

	// here we again get the handle to our current render target view so we can set it as the render target in the output merger stage of the pipeline
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(renderer->getDescriptionHeap()->GetCPUDescriptorHandleForHeapStart(), renderer->getFrameIndex(), renderer->getRtvDescriptorSize());

	// get a handle to the depth/stencil buffer
	CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(mpPipelineState->getDepthStencilDescriptorHeap()->GetCPUDescriptorHandleForHeapStart());

	// set the render target for the output merger stage (the output of the pipeline)
	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

	// Clear the render target by using the ClearRenderTargetView command
	commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	commandList->ClearDepthStencilView(mpPipelineState->getDepthStencilDescriptorHeap()->GetCPUDescriptorHandleForHeapStart(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

	commandList->SetGraphicsRootSignature(mpPipelineState->getRootSignature()); // set the root signature

	commandList->SetGraphicsRootConstantBufferView(0, mpPipelineState->getConstantBufferUploadHeap(frameIndex)->GetGPUVirtualAddress());

	/*
	// set constant buffer descriptor heap
	ID3D12DescriptorHeap* descriptorHeaps[] = { mpPipelineState->getConstantBufferDescriptorHeap(frameIndex) };
	commandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

	// set the root descriptor table 0 to the constant buffer descriptor heap
	commandList->SetGraphicsRootDescriptorTable(0, mpPipelineState->getConstantBufferDescriptorHeap(frameIndex)->GetGPUDescriptorHandleForHeapStart());
	*/

	// draw triangle
	commandList->RSSetViewports(1, &mViewport); // set the viewports
	commandList->RSSetScissorRects(1, &mScissorRect); // set the scissor rects
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); // set the primitive topology

	if (nullptr != mpModel)
	{
		for (int i = 0; i < mpModel->getMeshesCount(); ++i)
		{
			mpModel->drawMesh(i, mpCommandList);
		}
	}
	
	//commandList->IASetVertexBuffers(0, 1, &vertexBufferView); // set the vertex buffer (using the vertex buffer view)
	//commandList->DrawInstanced(3, 1, 0, 0); // finally draw 3 vertices (draw the triangle)

	// transition the "frameIndex" render target from the render target state to the present state. If the debug layer is enabled, you will receive a
	// warning if present is called on the render target when it's not in the present state
	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTarget[frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

	endRecording(frameIndex);
}
