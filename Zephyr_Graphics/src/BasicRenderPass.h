#ifndef BASIC_RENDER_PASS_H
#define BASIC_RENDER_PASS_H

#include "IRenderPass.h"
#include "RenderableModel.h"
#include <Camera.h>

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

				void setCamera(const Common::Camera& camera);
				std::shared_ptr<Common::Camera> initalizeCamera(const Common::Vector3f& cameraPos,
												const Common::Vector3f& cameraTarget,
												const Common::Vector3f& cameraUp,
												const float fov, // in radian
												const float nearClip,
												const float farClip,
												const int screenWidth,
												const int screenHeight);

				std::shared_ptr<Common::Camera> getCamera();
				const std::shared_ptr<Common::Camera> getCamera() const;
				void updateCameraMatrix();

				void setClearColor(float r, float g, float b, float a);
				virtual bool loadModel(const std::string& modelPath);

				virtual RenderableModel* getModel();

				virtual bool initialize();
				virtual void update(const int frameIndex, const double deltaTime);

			protected:
				std::string mModelPath;
				float clearColor[4];
				std::shared_ptr<RenderableModel> mpModel;
				ConstantBuffer mConstantBuffer;

				std::shared_ptr<Common::Camera> mpCamera;

				XMFLOAT4X4 cameraProjMat; // this will store our projection matrix
				XMFLOAT4X4 cameraViewMat; // this will store our view matrix

				/*XMFLOAT4 cameraPosition; // this is our cameras position vector
				XMFLOAT4 cameraTarget; // a vector describing the point in space our camera is looking at
				XMFLOAT4 cameraUp; // the worlds up vector*/

				XMFLOAT4X4 cube1WorldMat; // our first cubes world matrix (transformation matrix)
				XMFLOAT4X4 cube1RotMat; // this will keep track of our rotation for the first cube
				XMFLOAT4 cube1Position; // our first cubes position in space
		};
	}
}

#endif