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

	nana::nested_form nfm(form, nana::rectangle{ 10, 10, 100, 20 }, nana::form::appear::bald<>());	

	//nana::label label(form, nana::rectangle(0, 0, 100, 50));
	//label.caption("Hello Nana");
	//form.show();
	hwnd = reinterpret_cast<HWND>(form.native_handle());
	

	auto filePath = "..\\model\\Armadillo.ply";
	//auto filePath = "..\\model\\bunny.obj";
	//Common::Model model;
	//Common::MeshLoader::loadFile(filePath, &model);

	//auto mesh = model.getMesh(0);

	//Algorithm::TriDualGraph graph(&mesh);

	std::vector<std::vector<int>> input;
	input.push_back(std::vector<int>());
	input.back().push_back(10000);

	input.push_back(std::vector<int>());
	input.back().push_back(1);


	input.push_back(std::vector<int>());
	input.back().push_back(20750);


	input.push_back(std::vector<int>());
	input.back().push_back(5000);

	//graph.segment(input);

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

	engine.setupRenderPass(pRenderPass, "BasicRenderPass");

	//Define a directX rendering function
	form.draw_through([&]() mutable
	{
		if (engine.isRunning())
		{
			engine.render();
			RECT r;
			::GetClientRect(hwnd, &r);
			::InvalidateRect(hwnd, &r, FALSE);
		}
	});

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
	btn.show();

	nfm.show();
	form.show();
	nana::exec();
	
	return 0;
}
