#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers.
#endif

#include <windows.h>
#include <Zephyr_Graphics.h>
#include <iostream>
#include <tchar.h>
#include <TriDualGraph.h>
#include <MeshLoader.h>

using namespace Zephyr;

bool createWindow(
	HWND& hwnd,
	HINSTANCE hInstance,
	int ShowWnd,
	int width, int height,
	bool fullscreen,
	LPCTSTR windowName);

// main application loop
void mainloop(Graphics::GraphicsEngine& engine);

LRESULT CALLBACK WndProc(HWND hwnd,
	UINT msg,
	WPARAM wParam,
	LPARAM lParam)
{
	switch (msg)
	{

	case WM_KEYDOWN:
		if (wParam == VK_ESCAPE) {
			if (MessageBox(0, L"Are you sure you want to exit?",
				L"Really?", MB_YESNO | MB_ICONQUESTION) == IDYES)
				DestroyWindow(hwnd);
		}
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hwnd,
		msg,
		wParam,
		lParam);
}

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
	int width = 1920;
	int height = 1080;

	// is window full screen?
	bool fullScreen = false;

	auto filePath = "..\\model\\Armadillo.ply";
	//auto filePath = "..\\model\\bunny.obj";
	Common::Model model;
	Common::MeshLoader::loadFile(filePath, &model);

	auto mesh = model.getMesh(0);

	Algorithm::TriDualGraph graph(&mesh);

	std::vector<std::vector<int>> input;
	input.push_back(std::vector<int>());
	input.back().push_back(10000);
	/*input.back().push_back(2);
	input.back().push_back(3);
	input.back().push_back(4);*/

	input.push_back(std::vector<int>());
	input.back().push_back(1);
	/*input.back().push_back(101);
	input.back().push_back(102);
	input.back().push_back(103);*/

	input.push_back(std::vector<int>());
	input.back().push_back(20750);
	/*input.back().push_back(2001);
	input.back().push_back(2002);
	input.back().push_back(2003);*/

	input.push_back(std::vector<int>());
	input.back().push_back(5000);
	/*input.back().push_back(3001);
	input.back().push_back(3002);
	input.back().push_back(3003);*/

	graph.segment(input);

	// create window
	if (!createWindow(hwnd, hInstance, nCmdShow, width, height, fullScreen, windowName))
	{
		MessageBox(0, L"Window Initialization - Failed",
			L"Error", MB_OK);
		return 1;
	}

	Graphics::GraphicsEngine engine;
	// initialize direct3d
	if (!engine.initialize(width, height, hwnd, fullScreen))
	{
		MessageBox(0, L"Failed to initialize direct3d 12",
			L"Error", MB_OK);
		engine.cleanup();
		return 1;
	}

	// start the main loop
	mainloop(engine);
	
	return 0;
}

bool createWindow(HWND& hwnd, HINSTANCE hInstance, int ShowWnd, int width, int height, bool fullscreen, LPCTSTR windowName)
{
	if (fullscreen)
	{
		HMONITOR hmon = MonitorFromWindow(hwnd,
			MONITOR_DEFAULTTONEAREST);
		MONITORINFO mi = { sizeof(mi) };
		GetMonitorInfo(hmon, &mi);

		width = mi.rcMonitor.right - mi.rcMonitor.left;
		height = mi.rcMonitor.bottom - mi.rcMonitor.top;
	}

	WNDCLASSEX wc;

	wc.cbSize = sizeof(WNDCLASSEX);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = WndProc;
	wc.cbClsExtra = NULL;
	wc.cbWndExtra = NULL;
	wc.hInstance = hInstance;
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 2);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = windowName;
	wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	if (!RegisterClassEx(&wc))
	{
		MessageBox(NULL, L"Error registering class",
			L"Error", MB_OK | MB_ICONERROR);
		return false;
	}

	hwnd = CreateWindowEx(NULL,
		windowName,
		windowName,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT,
		width, height,
		NULL,
		NULL,
		hInstance,
		NULL);

	if (!hwnd)
	{
		MessageBox(NULL, L"Error creating window",
			L"Error", MB_OK | MB_ICONERROR);
		return false;
	}

	if (fullscreen)
	{
		SetWindowLong(hwnd, GWL_STYLE, 0);
	}

	ShowWindow(hwnd, ShowWnd);
	UpdateWindow(hwnd);

	return true;
}

void mainloop(Graphics::GraphicsEngine& engine)
{
	MSG msg;
	ZeroMemory(&msg, sizeof(MSG));

	while (engine.isRunning())
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				break;

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
			// run game code
			engine.render(); // execute the command queue (rendering the scene is the result of the gpu executing the command lists)
		}
	}
}
