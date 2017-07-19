#ifndef RENDER_PASS_H
#define RENDER_PASS_H

#include "stdfx.h"
#include "ShaderManager.h"
#include "CommandList.h"

namespace Zephyr
{
	namespace Graphics
	{
		class Renderer;
	}
}

namespace Zephyr
{
	namespace Graphics
	{
		class GraphicsEngine;
		class CommandList;

		class ZEPHYR_GRAPHICS_API IRenderPass : public CommandList
		{
			public:
				IRenderPass(const int frameBufferCount, GraphicsEngine* pEngine);
				virtual ~IRenderPass();

				virtual bool initialize() = 0;
				virtual void update(const int frameIndex, const double deltaTime) = 0;

			protected:
				virtual bool setupViewport(); 

			protected:
				D3D12_VIEWPORT mViewport; // area that output from rasterizer will be stretched to.
				D3D12_RECT mScissorRect; // the area to draw in. pixels outside that area will not be drawn onto
		};

	}
}
#endif
