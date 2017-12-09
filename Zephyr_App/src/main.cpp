#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>
#include <Zephyr_Graphics.h>
#include <iostream>
#include <tchar.h>
#include <TriDualGraph.h>
#include <IO/MeshLoader.h>
#include "UI.h"
#include "App.h"
#include <BasicRenderPass.h>

#include <nana/gui/wvl.hpp> 
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/timer.hpp>
#include <nana/gui/filebox.hpp>

#include <Mesh/OM_Mesh.h>

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

	App app;
	if (!app.initialize())
		return -1;

	app.start();
	
	return 0;
}
