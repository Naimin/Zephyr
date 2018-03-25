#ifndef ZEPHYR_GPU_RANDOM_H
#define ZEPHYR_GPU_RANDOM_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <vector>

namespace Zephyr
{
	namespace Random
	{
		template <typename T>
		struct prg_real
		{
			T a, b;

			__host__ __device__
				prg_real(T _a = 0.0f, T _b = 1.0f) : a(_a), b(_b) {};

			__host__ __device__
				float operator()(const unsigned int n) const
			{
				thrust::default_random_engine rng;
				thrust::uniform_real_distribution<T> dist(a, b);
				rng.discard(n);

				return dist(rng);
			}
		};

		template <typename T>
		struct prg_int
		{
			T a, b;

			__host__ __device__
				prg_int(T _a = 0, T _b = 100) : a(_a), b(_b) {};

			__host__ __device__
				float operator()(const unsigned int n) const
			{
				thrust::default_random_engine rng;
				thrust::uniform_int_distribution<T> dist(a, b);
				rng.discard(n);

				return dist(rng);
			}
		};

		template <typename T>
		static void generateRandomReal(thrust::device_vector<T>& output, const T minValue, const T maxValue, const int sequence)
		{
			thrust::counting_iterator<unsigned int> index_sequence_begin(sequence);
			Random::prg_real<T> randomRange(minValue, maxValue);

			thrust::transform(index_sequence_begin,
				index_sequence_begin + (int)output.size(),
				output.begin(),
				randomRange);
		}

		template <typename T>
		static void generateRandomInt(thrust::device_vector<T>& output, const T minValue, const T maxValue, int& sequence)
		{
			thrust::counting_iterator<unsigned int> index_sequence_begin(sequence);
			Random::prg_int<T> randomRange(minValue, maxValue);

			thrust::transform(index_sequence_begin,
				index_sequence_begin + (int)output.size(),
				output.begin(),
				randomRange);
			sequence += output.size();
		}
	}
}
/* // Usage Pattern
thrust::transform(index_sequence_begin,
				  index_sequence_begin + N,
				  numbers.begin(),
				  randomRange);

for (int i = 0; i < N; i++)
{
	std::cout << numbers[i] << std::endl;
}
*/

#endif