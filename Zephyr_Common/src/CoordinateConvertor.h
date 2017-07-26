#ifndef COORDINATE_CONVERTOR_H
#define COORDINATE_CONVERTOR_H

#include "stdfx.h"
#include <cmath>

namespace Zephyr
{
	namespace Common
	{
		const float PI = 3.14159265358979323846f;
		const float PI_OVER_180 = PI / 180.0f;
		const float OVER_180_PI = 180.0f / PI;

		struct ZEPHYR_COMMON_API CoordinateConvertor
		{
			// polar coordinate is R, Inclination angle, azimuth angle
			static Common::Vector3f CartesianToPolar(const Common::Vector3f& cartesianVec)
			{
				Common::Vector3f polar;

				polar.x() = cartesianVec.norm();
				polar.y() = std::acos(cartesianVec.z() / polar.x());
				polar.z() = std::atan2f(cartesianVec.y(), cartesianVec.x());

				return polar;
			}

			static Common::Vector3f PolarToCartesian(const Common::Vector3f& polarVec)
			{
				Common::Vector3f cartesian;

				cartesian.x() = polarVec.x() * std::cos(polarVec.y()) * std::sin(polarVec.z());
				cartesian.y() = polarVec.x() * std::sin(polarVec.y()) * std::sin(polarVec.z());
				cartesian.z() = polarVec.x() * std::cos(polarVec.z());

				return cartesian;
			}

			static float RadianToDegree(const float radian)
			{
				return radian * OVER_180_PI;
			}

			static float DegreeToRadian(const float degree)
			{
				return degree * PI_OVER_180;
			}
		};
	}
}

#endif
