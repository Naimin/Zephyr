#include "Camera.h"
#include "CoordinateConvertor.h"
#include <iostream>>

using namespace Zephyr::Common;

Camera::Camera()
{
}

Camera::~Camera()
{
}

void Camera::intialize(const Vector3f & cameraPos, const Vector3f & cameraTarget, const Vector3f & cameraUp, const float fov, const float nearClip, const float farClip, const float aspectRatio)
{
	mCameraPos = cameraPos.normalized();
	mCameraTarget = cameraTarget.normalized();
	mCameraUp = cameraUp.normalized();
	updateViewMatrix(mCameraPos, mCameraTarget, mCameraUp);

	mFOV = fov;
	mNearClip = nearClip;
	mFarClip = farClip;
	mAspectRatio = aspectRatio;
	updatePerspectiveMatrix(mFOV, mNearClip, mFarClip, mAspectRatio);

	mPolarCoord = CoordinateConvertor::CartesianToPolar(mCameraTarget - mCameraPos);
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
	mPolarCoord[0] += distance;
	mCameraPos = CoordinateConvertor::PolarToCartesian(mPolarCoord);
}

void Zephyr::Common::Camera::pan(const float deltaX, const float deltaY)
{
	auto sideDirection = getViewDirection().cross(mCameraUp);

	mCameraPos += sideDirection * deltaX;
	mCameraPos += mCameraUp * deltaY;
}

void Zephyr::Common::Camera::rotation(const float degreeX, const float degreeY)
{
	mPolarCoord[1] += CoordinateConvertor::DegreeToRadian(degreeY);
	mPolarCoord[2] += CoordinateConvertor::DegreeToRadian(degreeX);

	mCameraPos = CoordinateConvertor::PolarToCartesian(mPolarCoord);
}

Vector3f Camera::getViewDirection() const
{
	return (mCameraTarget - mCameraPos).normalized();
}
