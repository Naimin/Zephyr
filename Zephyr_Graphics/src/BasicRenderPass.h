#ifndef BASIC_RENDER_PASS_H
#define BASIC_RENDER_PASS_H

#include "IRenderPass.h"
#include "RenderableModel.h"

namespace Zephyr
{
	namespace Graphics
	{
		using namespace DirectX;

		class ZEPHYR_GRAPHICS_API BasicRenderPass : public IRenderPass
		{
			public:
				struct ConstantBuffer
				{
					XMFLOAT4X4 wvpMat;
				};

			public:
				BasicRenderPass(const int frameBufferCount, GraphicsEngine* pEngine);
				virtual ~BasicRenderPass();

				void setClearColor(float r, float g, float b, float a);
				virtual bool loadModel(const std::string& modelPath);

				virtual bool initialize();
				virtual void update(const int frameIndex);

			protected:
				std::string mModelPath;
				float clearColor[4];
				std::shared_ptr<RenderableModel> mpModel;
				ConstantBuffer mConstantBuffer;

				XMFLOAT4X4 cameraProjMat; // this will store our projection matrix
				XMFLOAT4X4 cameraViewMat; // this will store our view matrix

				XMFLOAT4 cameraPosition; // this is our cameras position vector
				XMFLOAT4 cameraTarget; // a vector describing the point in space our camera is looking at
				XMFLOAT4 cameraUp; // the worlds up vector

				XMFLOAT4X4 cube1WorldMat; // our first cubes world matrix (transformation matrix)
				XMFLOAT4X4 cube1RotMat; // this will keep track of our rotation for the first cube
				XMFLOAT4 cube1Position; // our first cubes position in space
		};
	}
}

#endif