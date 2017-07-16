#ifndef ZEPHYR_GRAPHICS_H
#define ZEPHYR_GRAPHICS_H

#include "stdfx.h"
#include "Renderer.h"
#include "ShaderManager.h"
#include "ResourceManager.h"

namespace Zephyr 
{
	namespace Graphics
	{
		class BasicRenderPass;

		class ZEPHYR_GRAPHICS_API GraphicsEngine
		{
			public:
				GraphicsEngine();
				virtual ~GraphicsEngine();

				bool initialize(unsigned int backBufferWidth, unsigned int backBufferHeight, HWND& hwnd, bool bFullScreen);
				bool setupRenderPass(IRenderPass* pRenderPass, const std::string& passName);
				void cleanup();

				void update();
				void render();

				void waitForPreviousFrame();
				
				
			public: // Accessor
				bool isRunning() const;
				std::shared_ptr<Renderer> getRenderer() const;
				std::shared_ptr<ShaderManager> getShaderManager() const;
				std::shared_ptr<ResourceManager> getResourceManager() const;
				unsigned int getWidth() const;
				unsigned int getHeight() const;

			private:
				unsigned int mBackBufferWidth, mBackBufferHeight;
				bool mbFullScreen;
				bool mbIsRunning;
				std::shared_ptr<Renderer> mpRenderer;
				std::shared_ptr<ShaderManager> mpShaderManager;
				std::shared_ptr<ResourceManager> mpResourceManager;

				std::vector<IRenderPass*> mRenderPasses;
				BasicRenderPass* mpRenderPass;
		};

	}
}

#endif