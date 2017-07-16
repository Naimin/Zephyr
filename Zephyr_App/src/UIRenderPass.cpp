#include "UIRenderPass.h"
#include <Renderer.h>
#include <Pipeline.h>
#include <Zephyr_Graphics.h>

using namespace Zephyr::Graphics;

Zephyr::UIRenderPass::UIRenderPass(const int frameBufferCount, GraphicsEngine* pEngine) : IRenderPass(frameBufferCount, pEngine)
{
	auto renderer = mpEngine->getRenderer();

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(renderer->getDescriptionHeap()->GetCPUDescriptorHandleForHeapStart(), renderer->getFrameIndex(), renderer->getRtvDescriptorSize());
	mpCommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
}

Zephyr::UIRenderPass::~UIRenderPass()
{
}

bool Zephyr::UIRenderPass::initialize()
{
	return true;
}

void Zephyr::UIRenderPass::update(const int frameIndex)
{
	startRecording(frameIndex);

	auto resourceManager = mpEngine->getResourceManager();
	// wait for the transfer to complete first 
	resourceManager->waitForPreviousTask();

	// Here go the actual rendering commands
	auto renderer = mpEngine->getRenderer();
	auto& commandList = mpCommandList;
	auto& renderTarget = renderer->getRenderTargets();

	// transition the "frameIndex" render target from the present state to the render target state so the command list draws to it starting from here
	//commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTarget[frameIndex], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));


	// here we again get the handle to our current render target view so we can set it as the render target in the output merger stage of the pipeline
	//CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(renderer->getDescriptionHeap()->GetCPUDescriptorHandleForHeapStart(), renderer->getFrameIndex(), renderer->getRtvDescriptorSize());
	//mpCommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

	//auto hr = renderer->getDevice()->GetDeviceRemovedReason();


	// get a handle to the depth/stencil buffer
	//CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(mpPipelineState->getDepthStencilDescriptorHeap()->GetCPUDescriptorHandleForHeapStart());

	// set the render target for the output merger stage (the output of the pipeline)
	

	// draw triangle
	//commandList->RSSetViewports(1, &mViewport); // set the viewports
	//commandList->RSSetScissorRects(1, &mScissorRect); // set the scissor rects

	//TwDrawContext(mpCommandList.get());
	//commandList->IASetVertexBuffers(0, 1, &vertexBufferView); // set the vertex buffer (using the vertex buffer view)
	//commandList->DrawInstanced(3, 1, 0, 0); // finally draw 3 vertices (draw the triangle)

	// transition the "frameIndex" render target from the render target state to the present state. If the debug layer is enabled, you will receive a
	// warning if present is called on the render target when it's not in the present state
	//commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTarget[frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
	
	endRecording(frameIndex);
}
