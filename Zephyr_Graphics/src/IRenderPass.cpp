#include "IRenderPass.h"
#include "Zephyr_Graphics.h"

Zephyr::Graphics::IRenderPass::IRenderPass(const int frameBufferCount, GraphicsEngine * pEngine) : CommandList(frameBufferCount, pEngine)
{
	setupViewport();
}

Zephyr::Graphics::IRenderPass::~IRenderPass()
{
}

bool Zephyr::Graphics::IRenderPass::setupViewport()
{
	// Fill out the Viewport
	mViewport.TopLeftX = 0;
	mViewport.TopLeftY = 0;
	mViewport.Width = (float)mpEngine->getWidth();
	mViewport.Height = (float)mpEngine->getHeight();
	mViewport.MinDepth = 0.0f;
	mViewport.MaxDepth = 1.0f;

	// Fill out a scissor rect
	mScissorRect.left = 0;
	mScissorRect.top = 0;
	mScissorRect.right = mpEngine->getWidth();
	mScissorRect.bottom = mpEngine->getHeight();
	
	return true;
}
