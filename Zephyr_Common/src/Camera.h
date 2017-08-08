#ifndef CAMERA_H
#define CAMERA_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API Camera
		{
			public:
				Camera();
				~Camera();

				void intialize(const Common::Vector3f& cameraPos,
					const Common::Vector3f& cameraTarget,
					const Common::Vector3f& cameraUp,
					const float fov, // in radian
					const float nearClip,
					const float farClip,
					const int screenWidth,
					const int screenHeight);

				void updatePerspectiveMatrix(const float fov, // in radian
											 const float aspectRatio,
											 const float nearClip, 
											 const float farClip);

				void updateViewMatrix(const Common::Vector3f& cameraPos,
									  const Common::Vector3f& cameraTarget,
									  const Common::Vector3f& cameraUp);

				void zoom(const float distance);
				void pan(const float deltaX, const float deltaY);
				void rotation(const float degreeX, const float degreeY);

				Common::Vector3f getViewDirection() const;

				Common::Vector3f getPickingRay(const int mouseX, const int mouseY) const;

			public:
				Common::Vector3f mCameraPos;
				Common::Vector3f mCameraTarget;
				Common::Vector3f mCameraUp;
				Common::Vector3f mPolarCoord; // Radius, Latitude, Azimuth

				float mFOV; // radian
				float mNearClip;
				float mFarClip;

				float mAspectRatio;

				Eigen::Matrix4f mPerspectiveMatrix;
				Eigen::Matrix4f mViewMatrix;

				int mScreenWidth;
				int mScreenHeight;
		};
	}
}

#endif
