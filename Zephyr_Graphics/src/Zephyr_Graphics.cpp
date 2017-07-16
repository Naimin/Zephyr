#include "Zephyr_Graphics.h"
#include <iostream>

#include <random>
#include "BasicRenderPass.h"
#include "TestRenderPass.h"

Zephyr::Graphics::GraphicsEngine::GraphicsEngine() : mpRenderer(new Renderer(this)), mpShaderManager(new ShaderManager)
{
}

Zephyr::Graphics::GraphicsEngine::~GraphicsEngine()
{
	cleanup();
}

bool Zephyr::Graphics::GraphicsEngine::initialize(unsigned int backBufferWidth, unsigned int backBufferHeight, HWND& hwnd, bool bFullScreen)
{
	if (nullptr == mpRenderer)
		return false;

	mbFullScreen = bFullScreen;
	mBackBufferWidth = backBufferWidth;
	mBackBufferHeight = backBufferHeight;

	auto success = mpRenderer->initialize(backBufferWidth, backBufferHeight, hwnd, bFullScreen);
	if (!success)
		return false;

	mpResourceManager.reset(new ResourceManager(this));
	success = mpResourceManager->initialize();
	if (!success)
		return false;

	success = setupRenderPass();
	if (!success)
		return false;

	return true;
}

bool Zephyr::Graphics::GraphicsEngine::setupRenderPass()
{
	if (nullptr == mpRenderer)
		return false;

	/*auto basic1 = new BasicRenderPass(mpRenderer.get(), mpShaderManager.get());
	basic1->setClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	mpRenderer->addRenderPass("BasicRenderPass", basic1);
	auto success = mpRenderer->enqueuRenderPass("BasicRenderPass", 0);

	auto basic2 = new BasicRenderPass(mpRenderer.get(), mpShaderManager.get());
	basic2->setClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	mpRenderer->addRenderPass("BasicRenderPass2", basic2);
	success = mpRenderer->enqueuRenderPass("BasicRenderPass2", 1);
	*/

	
	mpRenderPass = new BasicRenderPass(FRAME_BUFFER_COUNT, this);
	mpRenderPass->initialize();
	mpRenderPass->setClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	mpRenderer->addRenderPass("BasicRenderPass3", mpRenderPass);
	auto success = mpRenderer->enqueuRenderPass("BasicRenderPass3", 0);
	

	/*
	auto test = new TestRenderPass(FRAME_BUFFER_COUNT, this);
	test->setClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	mpRenderer->addRenderPass("TestRenderPass", test);
	success = mpRenderer->enqueuRenderPass("TestRenderPass", 1);
	*/
	return success;
}

void Zephyr::Graphics::GraphicsEngine::cleanup()
{
	// clear the mpRenderer;
	if(nullptr != mpRenderer)
		mpRenderer.reset();
}

void Zephyr::Graphics::GraphicsEngine::update()
{
}

void Zephyr::Graphics::GraphicsEngine::render()
{
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	//mpRenderPass->setClearColor(dist(e2), dist(e2), dist(e2), 1.0f);

	mpRenderer->render();
}

void Zephyr::Graphics::GraphicsEngine::waitForPreviousFrame()
{
}

bool Zephyr::Graphics::GraphicsEngine::isRunning() const
{
	// clear the mpRenderer;
	if (nullptr != mpRenderer)
		return mpRenderer->isRunning();
	else
		return false;
}

std::shared_ptr<Zephyr::Graphics::Renderer> Zephyr::Graphics::GraphicsEngine::getRenderer() const
{
	return mpRenderer;
}

std::shared_ptr<Zephyr::Graphics::ShaderManager> Zephyr::Graphics::GraphicsEngine::getShaderManager() const
{
	return mpShaderManager;
}

std::shared_ptr<Zephyr::Graphics::ResourceManager> Zephyr::Graphics::GraphicsEngine::getResourceManager() const
{
	return mpResourceManager;
}

unsigned int Zephyr::Graphics::GraphicsEngine::getWidth() const
{
	return mBackBufferWidth;
}

unsigned int Zephyr::Graphics::GraphicsEngine::getHeight() const
{
	return mBackBufferHeight;
}
