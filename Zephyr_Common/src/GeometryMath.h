#ifndef GEOMETRY_MATH_H
#define GEOMETRY_MATH_H

#include <Eigen/Eigen>

// this interface so we can easily switch out the math library if we want to
namespace Zephyr
{
	namespace Common
	{

		 typedef Eigen::Vector3f Vector3f;

	}
}

#endif
