#include "LNormUtil.h"

using namespace Zephyr::Common;

float LNormUtil::L1Norm(const Vector3f vec)
{
	float result = 0;
	for (int i = 0; i < 3; ++i)
	{
		result += std::abs(vec[i]);
	}
	return result;
}

float LNormUtil::L2Norm(const Vector3f vec)
{
	float result = 0;
	for (int i = 0; i < 3; ++i)
	{
		result += std::abs(vec[1]) * std::abs(vec[1]);
	}
	return std::sqrt(result);
}

float LNormUtil::LInfinityNorm(const Vector3f vec)
{
	float result = 0;
	for (int i = 0; i < 3; ++i)
	{
		// get the max component of the vector
		result = std::max(std::abs(vec[i]), result);
	}
	return result;
}