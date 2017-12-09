#ifndef ZEPHYR_APP_H
#define ZEPHYR_APP_H

#include <Zephyr_Graphics.h>
#include <Camera.h>

namespace Zephyr
{
	class UI;
	class Common::Camera;

	class App
	{
		public:
			App(const std::string& windowTitle, unsigned int width = 800, unsigned int height = 600, bool bFullScreen = false);
			virtual ~App();
		
			bool initialize();
			bool start();
			
			std::shared_ptr<Graphics::GraphicsEngine> getGraphicsEngine();
			std::shared_ptr<UI> getUI();
			Common::Camera& getCamera();
			HWND& getHwnd();
			unsigned int getWidth() const;
			unsigned int getHeight() const;

		private:
			std::shared_ptr<Graphics::GraphicsEngine> mpEngine;
			std::shared_ptr<UI> mpUI;
			Common::Camera mCamera;
			HWND mHwnd;
			unsigned int mWidth;
			unsigned int mHeight;
			bool mbFullScreen;
	};
}

#endif