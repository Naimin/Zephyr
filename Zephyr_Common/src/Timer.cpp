#include "Timer.h"

using namespace std::chrono;

Zephyr::Common::Timer::Timer() : mStartTime(high_resolution_clock::now()), mPrevDeltaTime(mStartTime)
{
}

Zephyr::Common::Timer::~Timer()
{
}

double Zephyr::Common::Timer::getElapsedTime()
{
	high_resolution_clock::time_point now = high_resolution_clock::now();
	return getTimeDifference(mStartTime, now);
}

double Zephyr::Common::Timer::getDeltaTime()
{
	high_resolution_clock::time_point now = high_resolution_clock::now();
	auto deltaTime = getTimeDifference(mPrevDeltaTime, now);
	mPrevDeltaTime = now; // update prevDeltaTime

	return deltaTime;
}

double Zephyr::Common::Timer::getTimeDifference(const high_resolution_clock::time_point & startTime, const high_resolution_clock::time_point & endTime)
{
	duration<double> time_span = duration_cast<duration<double>>(endTime - startTime);
	return time_span.count();
}
