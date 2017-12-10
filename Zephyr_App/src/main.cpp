#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>
#include "App.h"

using namespace Zephyr;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	// title of the window
	std::string windowTitle = "Zephyr_App";

	// width and height of the window
	unsigned int width = 800;
	unsigned int height = 600;
	bool bFullScreen = false;

	App app(windowTitle, width, height, bFullScreen);
	if (!app.initialize())
		return -1;

	app.start();
	
	return 0;
}
