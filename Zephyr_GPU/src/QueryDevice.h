#ifndef ZEPHYR_GPU_QUERY_DEVICE_H
#define ZEPHYR_GPU_QUERY_DEVICE_H

#include "stdfx.h"
#include <vector>

namespace Zephyr
{
	namespace GPU
	{
		class ZEPHYR_GPU_API QueryDevice
		{
			public:
				static int getDeviceCount();
				static std::string getDeviceName(const int gpuId);
				// MP stats
				static int getMultiProcessorCount(const int gpuId);
				static int getMaxThreadPerMultiProcessor(const int gpuId);
				static int getSharedMemPerMP(const int gpuId);
				// Grid stats
				static std::vector<int> getMaxGridSize(const int gpuId);
				// Block stats
				static std::vector<int> getMaxBlockSize(const int gpuId);
				static int getMaxThreadPerBlock(const int gpuId);
				static int getSharedMemPerBlock(const int gpuId);
				// GPU memory
				static int getTotalDeviceMem(const int gpuId);
				static int getTotalConstMem(const int gpuId);
				static size_t getAvailableMem();

				static void printQuery();

				// Util
				static int computeOptimalBlockCount(const int workSize, const int threadPerBlock, const int gpuId);
		};
	}
}

#endif