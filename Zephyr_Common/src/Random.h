#ifndef COMMON_RANDOM_H
#define COMMON_RANDOM_H

#include "stdfx.h"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

namespace Zephyr
{
	namespace Common
	{
		typedef boost::mt19937 RNGType;

		class ZEPHYR_COMMON_API RandomGenerator
		{
		public:
			RandomGenerator(int min, int max, int seed);

			virtual ~RandomGenerator();

			int next();

		private:
			RNGType mRng;
			boost::uniform_int<> mRange;
			boost::variate_generator< RNGType, boost::uniform_int<> > mDice;
		};
	}
}

#endif