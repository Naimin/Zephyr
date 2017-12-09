#include "App.h"
#include "UI.h"
#include "AppEvents.h"

#include <windows.h>
#include <Zephyr_Graphics.h>
#include <BasicRenderPass.h>
#include <nana/gui/timer.hpp>

using namespace Zephyr;

Zephyr::App::App(const std::string& windowTitle, unsigned int width, unsigned int height, bool bFullScreen) : mWidth(width), mHeight(height), mbFullScreen(bFullScreen), mHwnd(NULL)
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	// setup UI
	mpUI.reset(new UI(windowTitle, this));
}

Zephyr::App::~App()
{
}

bool Zephyr::App::initialize()
{
	mpEngine.reset(new Graphics::GraphicsEngine());
	// initialize direct3d
	if (!mpEngine->initialize(mWidth, mHeight, mHwnd, mbFullScreen))
	{
		MessageBox(0, L"Failed to initialize direct3d 12",
			L"Error", MB_OK);
		mpEngine->cleanup();
		return false;
	}

	auto pRenderPass = new Graphics::BasicRenderPass(3, mpEngine.get());
	pRenderPass->initialize();
	pRenderPass->setClearColor(0.2f, 0.2f, 0.2f, 1.0f);

	// Initalize the renderpass camera, and get reference to it
	mCamera = pRenderPass->initalizeCamera(
		Common::Vector3f(0.0f, 2.0f, -200.0f),
		Common::Vector3f(0, 0, 0),
		Common::Vector3f(0, 1.0f, 0),
		45.0f*(3.14f / 180.0f),
		0.1f,
		1000.0f,
		(int)mWidth,
		(int)mHeight);

	mpEngine->setupRenderPass(pRenderPass, "BasicRenderPass");

	if (!mpUI->initialize())
	{
		MessageBox(0, L"Failed to initialize UI framework",
			L"Error", MB_OK);
		mpEngine->cleanup();
		return false;
	}
	
	// setup buttons event
	AppEvents appEvents(this, mpUI.get());
	if (!appEvents.initialize())
	{
		MessageBox(0, L"Failed to initialize App Events",
			L"Error", MB_OK);
		mpEngine->cleanup();
		return false;
	}

	return true;
}

bool Zephyr::App::start()
{
	// Draw Through / Render Events
	//Define a directX rendering function
	mpUI->getForm()->draw_through([&]() mutable
	{
		if (mpEngine->isRunning())
		{
			mpEngine->render();
		}
	});

	nana::timer tmr;
	tmr.elapse([&] {
		RECT r;
		::GetClientRect(mHwnd, &r);
		::InvalidateRect(mHwnd, &r, FALSE);
	});
	tmr.interval(1); // this is the UI refresh rate
	tmr.start();

	mpUI->show();

	// let nana take over control flow
	nana::exec();

	return true;
}

std::shared_ptr<Graphics::GraphicsEngine> Zephyr::App::getGraphicsEngine()
{
	return mpEngine;
}

std::shared_ptr<UI> Zephyr::App::getUI()
{
	return mpUI;
}

Common::Camera & Zephyr::App::getCamera()
{
	return mCamera;
}

HWND & Zephyr::App::getHwnd()
{
	return mHwnd;
}

unsigned int Zephyr::App::getWidth() const
{
	return mWidth;
}

unsigned int Zephyr::App::getHeight() const
{
	return mHeight;
}
