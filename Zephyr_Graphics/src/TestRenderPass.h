#ifndef TEST_RENDER_PASS_H
#define TEST_RENDER_PASS_H

#include "IRenderPass.h"

namespace Zephyr
{
	namespace Graphics
	{
		class TestRenderPass : public IRenderPass
		{
			public:

				TestRenderPass(const int frameBufferCount, GraphicsEngine* pEngine);
				virtual ~TestRenderPass();

				void setClearColor(float r, float g, float b, float a);

				virtual bool initialize();
				virtual void update(const int frameIndex);

			protected:
				float clearColor[4];
		};
	}
}

#endif