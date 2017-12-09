#include "TestRenderPass.h"
#include "Renderer.h"
#include "Zephyr_Graphics.h"
#include <Primitive/Vertex.h>

Zephyr::Graphics::TestRenderPass::TestRenderPass(const int frameBufferCount, GraphicsEngine* pEngine) : IRenderPass(frameBufferCount, pEngine)
{
	setClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

Zephyr::Graphics::TestRenderPass::~TestRenderPass()
{
}

void Zephyr::Graphics::TestRenderPass::setClearColor(float r, float g, float b, float a)
{
	clearColor[0] = r;
	clearColor[1] = b;
	clearColor[2] = g;
	clearColor[3] = a;
}

bool Zephyr::Graphics::TestRenderPass::initialize()
{
	return true;
}

void Zephyr::Graphics::TestRenderPass::update(const int frameIndex, const double deltaTime)
{
	startRecording(frameIndex);

	auto resourceManager = mpEngine->getResourceManager();
	// wait for the transfer to complete first 
	resourceManager->waitForPreviousTask();

	// Here go the actual rendering commands
	auto renderer = mpEngine->getRenderer();
	auto& commandList = mpCommandList;
	auto& renderTarget = renderer->getRenderTargets();

	endRecording(frameIndex);
}
