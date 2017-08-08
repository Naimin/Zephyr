#include "Camera.h"
#include "CoordinateConvertor.h"
#include <iostream>

using namespace Zephyr::Common;

Camera::Camera()
{
}

Camera::~Camera()
{
}

void Camera::intialize(const Vector3f & cameraPos, const Vector3f & cameraTarget, const Vector3f & cameraUp, const float fov, const float nearClip, const float farClip, const int screenWidth, const int screenHeight)
{
	mCameraPos = cameraPos.normalized();
	mCameraTarget = cameraTarget.normalized();
	mCameraUp = cameraUp.normalized();
	updateViewMatrix(mCameraPos, mCameraTarget, mCameraUp);

	mFOV = fov;
	mNearClip = nearClip;
	mFarClip = farClip;
	mAspectRatio = screenWidth / (float)screenHeight;
	updatePerspectiveMatrix(mFOV, mNearClip, mFarClip, mAspectRatio);

	mScreenWidth = screenWidth;
	mScreenHeight = screenHeight;

	mPolarCoord = CoordinateConvertor::CartesianToPolar(mCameraTarget - mCameraPos);
}

void Camera::updatePerspectiveMatrix(const float fov, const float aspectRatio, const float nearClip, const float farClip)
{
	float yScale = 1.0f / std::tanf(fov / 2); // cot( fovY / 2)
	float xScale = yScale / aspectRatio;
	float fRange = farClip / (farClip - nearClip);
	/* 
	xScale     0          0               0
		0    yScale       0               0
		0      0       zf / (zf - zn)     1
		0      0     -zn*zf / (zf - zn)   0
	where:
	yScale = cot(fovY / 2)
	xScale = yScale / aspect ratio
	*/

	mPerspectiveMatrix.Zero();

	mPerspectiveMatrix(0, 0) = xScale;
	mPerspectiveMatrix(1, 1) = yScale;
	mPerspectiveMatrix(2, 2) = fRange;
	mPerspectiveMatrix(2, 3) = 1.0f;
	mPerspectiveMatrix(3, 2) = -fRange * nearClip;
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

	Vector3f negCameraPos = -1 * cameraPos;

	mViewMatrix.setIdentity();

	mViewMatrix(0, 0) = xAxis.x();
	mViewMatrix(1, 0) = xAxis.y();
	mViewMatrix(2, 0) = xAxis.z();
	mViewMatrix(3, 0) = xAxis.dot(negCameraPos);

	mViewMatrix(0, 1) = yAxis.x();
	mViewMatrix(1, 1) = yAxis.y();
	mViewMatrix(2, 1) = yAxis.z();
	mViewMatrix(3, 1) = yAxis.dot(negCameraPos);

	mViewMatrix(0, 2) = zAxis.x();
	mViewMatrix(1, 2) = zAxis.y();
	mViewMatrix(2, 2) = zAxis.z();
	mViewMatrix(3, 2) = zAxis.dot(negCameraPos);

	mViewMatrix.transposeInPlace();
}

void Camera::zoom(const float distance)
{
	mPolarCoord[0] += distance;
	mCameraPos = CoordinateConvertor::PolarToCartesian(mPolarCoord);
}

void Camera::pan(const float deltaX, const float deltaY)
{
	auto sideDirection = getViewDirection().cross(mCameraUp);

	mCameraPos += sideDirection * deltaX;
	mCameraPos += mCameraUp * deltaY;
}

void Camera::rotation(const float degreeX, const float degreeY)
{
	mPolarCoord[1] += CoordinateConvertor::DegreeToRadian(degreeY);
	mPolarCoord[2] += CoordinateConvertor::DegreeToRadian(degreeX);

	mCameraPos = CoordinateConvertor::PolarToCartesian(mPolarCoord);
}

Vector3f Camera::getViewDirection() const
{
	return (mCameraTarget - mCameraPos).normalized();
}

Vector3f Camera::getPickingRay(const int mouseX, const int mouseY) const
{
	// reference: http://www.rastertek.com/dx11tut47.html
	
	Vector2f p(mouseX, mouseY);

	float X = ((2.0f * (float)mouseX) / (float)mScreenWidth) - 1.0f;
	float Y = (((2.0f * (float)mouseY) / (float)mScreenHeight) - 1.0f) * -1.0f;

	X /= mPerspectiveMatrix(0, 0);
	Y /= mPerspectiveMatrix(1, 1);

	Vector4f origin = Vector4f(X, Y, 0, 1);
	Vector4f faraway = Vector4f(X, Y, 1, 1);

	Eigen::Matrix4f inverseView = mViewMatrix.inverse();

	Vector3f rayDirection;
	rayDirection.x() = (X * inverseView(0, 0)) + (Y * inverseView(1, 0)) + inverseView(2, 0);
	rayDirection.y() = (X * inverseView(0, 1)) + (Y * inverseView(1, 1)) + inverseView(2, 1);
	rayDirection.z() = (X * inverseView(0, 2)) + (Y * inverseView(1, 2)) + inverseView(2, 2);

	rayDirection.normalize();

	return rayDirection;
}