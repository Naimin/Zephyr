#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>
#include <Zephyr_Graphics.h>
#include <iostream>
#include <tchar.h>
#include <TriDualGraph.h>
#include <MeshLoader.h>
#include "UI.h"
#include "App.h"
#include <BasicRenderPass.h>

#include <nana/gui/wvl.hpp> 
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/timer.hpp>
#include <nana/gui/filebox.hpp>

using namespace Zephyr;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	// Handle to the window
	HWND hwnd = NULL;
	// name of the window (not the title)
	auto windowName = L"Zephyr_App";
	// title of the window
	auto windowTitle = L"Zephyr_App";

	// width and height of the window
	unsigned int width = 800;
	unsigned int height = 600;

	// is window full screen?
	bool fullScreen = false;

	nana::form form(nana::rectangle{ 0, 0, width, height });
	form.caption(windowTitle);

	nana::nested_form nfm(form, nana::rectangle{ 10, 10, 100, 50 }, nana::form::appear::bald<>());

	//nana::label label(form, nana::rectangle(0, 0, 100, 50));
	//label.caption("Hello Nana");
	//form.show();
	hwnd = reinterpret_cast<HWND>(form.native_handle());

	Graphics::GraphicsEngine engine;
	// initialize direct3d
	if (!engine.initialize(width, height, hwnd, fullScreen))
	{
		MessageBox(0, L"Failed to initialize direct3d 12",
			L"Error", MB_OK);
		engine.cleanup();
		return 1;
	}

	auto pRenderPass = new Graphics::BasicRenderPass(3, &engine);
	pRenderPass->initialize();
	pRenderPass->setClearColor(0.2f, 0.2f, 0.2f, 1.0f);

	// Initalize the renderpass camera, and get reference to it
	auto& camera = pRenderPass->initalizeCamera(
		Common::Vector3f(0.0f, 2.0f, -200.0f),
		Common::Vector3f(0, 0, 0),
		Common::Vector3f(0, 1.0f, 0),
		45.0f*(3.14f / 180.0f),
		0.1f,
		1000.0f,
		width,
		height);

	engine.setupRenderPass(pRenderPass, "BasicRenderPass");

	// Draw Through / Render Events
	//Define a directX rendering function
	form.draw_through([&]() mutable
	{
		if (engine.isRunning())
		{
			engine.render();
		}
	});

	// Load Model Events
	// load the user specified model
	nana::button btn(nfm, nana::rectangle{ 0, 0, 100, 20 });
	btn.caption(L"Load");
	btn.events().click([&] {
		nana::filebox fb(nfm, true);
		fb.add_filter("Model File", "*.obj;*.ply;*.fbx");
		fb.add_filter("All Files", "*.*");

		if (fb())
		{
			auto modelPath = fb.file();
			std::cout << modelPath << std::endl;

			pRenderPass->loadModel(modelPath);
		}
	});

	// Segmentation Events
	nana::button segmentBtn(nfm, nana::rectangle{ 0, 30, 100, 20 });
	segmentBtn.caption(L"Segment");
	segmentBtn.events().click([&] {
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		auto mesh = pModel->getMesh(0);
		Algorithm::TriDualGraph graph(&mesh);

		std::vector<std::vector<int>> input;
		input.push_back(std::vector<int>());
		input.back().push_back(10000);

		input.push_back(std::vector<int>());
		input.back().push_back(1);

		input.push_back(std::vector<int>());
		input.back().push_back(20750);

		input.push_back(std::vector<int>());
		input.back().push_back(5000);

		graph.segment(input);
	});

	// Mouse Events

	float zoom = 0;
	Eigen::Vector2i mousePosition;

	form.events().mouse_move([&](const nana::arg_mouse& mouseEvent)
	{
		int deltaX = mouseEvent.pos.x - mousePosition[0];
		int deltaY = mouseEvent.pos.y - mousePosition[1];

		// hold Mid mouse for zoom
		if (mouseEvent.mid_button)
		{
			camera.zoom((float)-deltaY);
		}

		if (mouseEvent.right_button)
		{
			camera.rotation((float)deltaX * 0.75f, (float)deltaY * 0.75f);
		}

		mousePosition[0] = mouseEvent.pos.x;
		mousePosition[1] = mouseEvent.pos.y;
	});

	form.events().click([&](const nana::arg_click& clickEvent)
	{
		if (clickEvent.mouse_args->is_left_button())
		{
			auto pickingRay = camera.getPickingRay(clickEvent.mouse_args->pos.x, clickEvent.mouse_args->pos.y);
		}
	});

	// one notch of mouse wheel delta is 120
	const float MOUSE_WHEEL_DELTA = 120.0f;
	form.events().mouse_wheel([&](const nana::arg_wheel& mouseWheelEvent)
	{
		auto distance = (mouseWheelEvent.distance / MOUSE_WHEEL_DELTA) * 10;
		distance = mouseWheelEvent.upwards ? distance : -distance;
		
		camera.zoom(distance);
	});

	nana::timer tmr;
	tmr.elapse([hwnd] {
		RECT r;
		::GetClientRect(hwnd, &r);
		::InvalidateRect(hwnd, &r, FALSE);
	});

	tmr.interval(1);
	tmr.start();

	nfm.show();
	form.show();
	nana::exec();
	
	return 0;
}
