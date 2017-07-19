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
					const float aspectRatio);

				void updatePerspectiveMatrix(const float fov, // in radian
											 const float aspectRatio,
											 const float nearClip, 
											 const float farClip);

				void updateViewMatrix(const Common::Vector3f& cameraPos,
									  const Common::Vector3f& cameraTarget,
									  const Common::Vector3f& cameraUp);

				void zoom(const float distance);

				Common::Vector3f getViewDirection() const;

			public:
				Common::Vector3f mCameraPos;
				Common::Vector3f mCameraTarget;
				Common::Vector3f mCameraUp;
				float mFOV; // radian
				float mNearClip;
				float mFarClip;

				float mAspectRatio;

				Eigen::Matrix4f mPerspectiveMatrix;
				Eigen::Matrix4f mViewMatrix;
		};
	}
}

#endif