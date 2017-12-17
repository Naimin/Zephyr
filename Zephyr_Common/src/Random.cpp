#include "Random.h"
#include <random>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

using namespace Zephyr;
using namespace Zephyr::Common;

RandomGenerator::RandomGenerator(int min, int max, int seed) :mRng(seed), mRange(min, max), mDice(mRng, mRange)
{
}

RandomGenerator::~RandomGenerator()
{

}

int RandomGenerator::next()
{
	return mDice();
}

