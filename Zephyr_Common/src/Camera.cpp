#include "Camera.h"

using namespace Zephyr::Common;

Camera::Camera()
{
}

Camera::~Camera()
{
}

void Camera::intialize(const Vector3f & cameraPos, const Vector3f & cameraTarget, const Vector3f & cameraUp, const float fov, const float nearClip, const float farClip, const float aspectRatio)
{
	mCameraPos = cameraPos;
	mCameraTarget = cameraTarget;
	mCameraUp = cameraUp;
	updateViewMatrix(mCameraPos, mCameraTarget, mCameraUp);

	mFOV = fov;
	mNearClip = nearClip;
	mFarClip = farClip;
	mAspectRatio = aspectRatio;
	updatePerspectiveMatrix(mFOV, mNearClip, mFarClip, mAspectRatio);
}

void Camera::updatePerspectiveMatrix(const float fov, const float aspectRatio, const float nearClip, const float farClip)
{
	float yScale = 1.0f / std::tanf(fov / 2); // cot( fovY / 2)
	float xScale = yScale / aspectRatio;

	/* 
	xScale     0          0               0
		0    yScale       0               0
		0      0       zf / (zf - zn)     1
		0      0     -zn*zf / (zf - zn)   0
	where:
	yScale = cot(fovY / 2)
	xScale = yScale / aspect ratio
	*/

	// Eigen is column-major vs directx row-major
	mPerspectiveMatrix.Zero();

	mPerspectiveMatrix(0, 0) = xScale;
	mPerspectiveMatrix(1, 1) = yScale;
	mPerspectiveMatrix(2, 2) = farClip / (farClip - nearClip);
	mPerspectiveMatrix(3, 2) = 1.0f;
	mPerspectiveMatrix(2, 3) = -nearClip * farClip / (farClip - nearClip);

	// now it is in Row-Major
	mViewMatrix.transposeInPlace();
}

void Camera::updateViewMatrix(const Vector3f & cameraPos, const Vector3f & cameraTarget, const Vector3f & cameraUp)
{
	/*
	zaxis = normal(cameraTarget - cameraPosition)
	xaxis = normal(cross(cameraUpVector, zaxis))
	yaxis = cross(zaxis, xaxis)
	
	 xaxis.x           yaxis.x           zaxis.x          0
	 xaxis.y           yaxis.y           zaxis.y          0
	 xaxis.z           yaxis.z           zaxis.z          0
	-dot(xaxis, cameraPosition)  -dot(yaxis, cameraPosition)  -dot(zaxis, cameraPosition)  1
	*/

	Vector3f zAxis = (cameraTarget - cameraPos).normalized();
	Vector3f xAxis = cameraUp.cross(zAxis).normalized();
	Vector3f yAxis = zAxis.cross(xAxis);

	mViewMatrix.setIdentity();

	mViewMatrix(0, 0) = xAxis.x();
	mViewMatrix(1, 0) = xAxis.y();
	mViewMatrix(2, 0) = xAxis.y();
	mViewMatrix(3, 0) = -(xAxis.dot(cameraPos));

	mViewMatrix(0, 1) = yAxis.x();
	mViewMatrix(1, 1) = yAxis.y();
	mViewMatrix(2, 1) = yAxis.y();
	mViewMatrix(3, 1) = -(yAxis.dot(cameraPos));

	mViewMatrix(0, 2) = zAxis.x();
	mViewMatrix(1, 2) = zAxis.y();
	mViewMatrix(2, 2) = zAxis.y();
	mViewMatrix(3, 2) = -(zAxis.dot(cameraPos));

	// now it is in Row-Major
	mViewMatrix.transposeInPlace();
}

void Zephyr::Common::Camera::zoom(const float distance)
{
	auto zoomAmount = distance * getViewDirection();

	mCameraPos += zoomAmount;
	mCameraTarget += zoomAmount;
}

Vector3f Camera::getViewDirection() const
{
	return (mCameraTarget - mCameraPos).normalized();
}
