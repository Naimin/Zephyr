#ifndef COMMON_TIMER_H
#define COMMON_TIMER_H

#include "stdfx.h"
#include <chrono>

namespace Zephyr
{
	namespace Common
	{
		class ZEPHYR_COMMON_API Timer
		{
			public:
				Timer();
				virtual ~Timer();

				double getElapsedTime(); // Time since creation of the Timer
				double getDeltaTime(); // Time since last call to this function

				double getTimeDifference(const std::chrono::high_resolution_clock::time_point& startTime,
										 const std::chrono::high_resolution_clock::time_point& endTime);

			private:
				std::chrono::high_resolution_clock::time_point mStartTime;
				std::chrono::high_resolution_clock::time_point mPrevDeltaTime;
		};
	}
}

#endif
