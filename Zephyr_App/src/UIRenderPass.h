#ifndef UI_RENDER_PASS_H
#define UI_RENDER_PASS_H

#include "IRenderPass.h"

namespace Zephyr
{
	class UIRenderPass : public Graphics::IRenderPass
	{
	public:
		UIRenderPass(const int frameBufferCount, Graphics::GraphicsEngine* pEngine);
		virtual ~UIRenderPass();

		virtual bool initialize();
		virtual void update(const int frameIndex);
	};
}

#endif