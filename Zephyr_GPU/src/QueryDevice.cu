#include "QueryDevice.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

using namespace Zephyr;
using namespace Zephyr::GPU;

int Zephyr::GPU::QueryDevice::getDeviceCount()
{
	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	return nDevices;
}

std::string Zephyr::GPU::QueryDevice::getDeviceName(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.name;
}

int Zephyr::GPU::QueryDevice::getMultiProcessorCount(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.multiProcessorCount;
}

int Zephyr::GPU::QueryDevice::getMaxThreadPerMultiProcessor(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.maxThreadsPerMultiProcessor;
}

int Zephyr::GPU::QueryDevice::getSharedMemPerMP(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.sharedMemPerMultiprocessor;
}

std::vector<int> Zephyr::GPU::QueryDevice::getMaxGridSize(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);
	auto maxGridSize = prop.maxGridSize;

	std::vector<int> result(3);
	result[0] = maxGridSize[0];
	result[1] = maxGridSize[1];
	result[2] = maxGridSize[2];

	return result;
}

std::vector<int> Zephyr::GPU::QueryDevice::getMaxBlockSize(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);
	auto maxBlockSize = prop.maxThreadsDim;

	std::vector<int> result(3);
	result[0] = maxBlockSize[0];
	result[1] = maxBlockSize[1];
	result[2] = maxBlockSize[2];

	return result;
}

int Zephyr::GPU::QueryDevice::getMaxThreadPerBlock(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.maxThreadsPerMultiProcessor;
}

int Zephyr::GPU::QueryDevice::getSharedMemPerBlock(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.sharedMemPerBlock;
}

int Zephyr::GPU::QueryDevice::getTotalDeviceMem(const int gpuId)
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);

	return total;
}

int Zephyr::GPU::QueryDevice::getTotalConstMem(const int gpuId)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, gpuId);

	return prop.totalConstMem;
}

size_t Zephyr::GPU::QueryDevice::getAvailableMem()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);

	return free;
}

void Zephyr::GPU::QueryDevice::printQuery()
{
	int deviceCount = QueryDevice::getDeviceCount();
	std::cout << "Device Count: " << deviceCount << std::endl;

	for (int i = 0; i < deviceCount; ++i)
	{
		std::cout << "*************** Device " << i << " ***************" << std::endl;
		std::cout << "Device Name: " << QueryDevice::getDeviceName(i).c_str() << std::endl;
		std::cout << "MP Count: " << QueryDevice::getMultiProcessorCount(i) << std::endl;
		std::cout << "Thread per MP: " << QueryDevice::getMaxThreadPerMultiProcessor(i) << std::endl;
		std::cout << "Shared Mem per MP: " << QueryDevice::getSharedMemPerMP(i) << std::endl;

		auto gridSize = QueryDevice::getMaxGridSize(i);
		std::cout << "Grid Dim: " << gridSize[0] << " , " << gridSize[1] << ", " << gridSize[2] << std::endl;
		auto blockSize = QueryDevice::getMaxBlockSize(i);
		std::cout << "Block Dim: " << blockSize[0] << " , " << blockSize[1] << ", " << blockSize[2] << std::endl;
		std::cout << "Thread per block: " << QueryDevice::getMaxThreadPerBlock(i) << std::endl;
		std::cout << "Shared Mem per block: " << QueryDevice::getSharedMemPerBlock(i) << std::endl;

		std::cout << "Total Global Mem: " << QueryDevice::getTotalDeviceMem(i) << std::endl;
		std::cout << "Total Const Mem: " << QueryDevice::getTotalConstMem(i) << std::endl;
		std::cout << "Total Available Mem: " << QueryDevice::getAvailableMem() << std::endl;
		std::cout << "*************** End Device " << i << " ***************" << std::endl;
	}
}

int Zephyr::GPU::QueryDevice::computeOptimalBlockCount(const int workSize, const int threadPerBlock, const int gpuId)
{
	// check total number of thread in MP
	int numOfMP = GPU::QueryDevice::getMultiProcessorCount(gpuId);
	int maxThreadPerMP = GPU::QueryDevice::getMaxThreadPerMultiProcessor(gpuId);
	// max block per MP can't be more than grid.x (hack atm)
	int maxBlockPerMP = std::min(maxThreadPerMP / threadPerBlock, getMaxGridSize(gpuId)[0]);

	int maxBlockOneIter = numOfMP * maxBlockPerMP;
	// check if there is enough work in the first place
	int blocksRequired = ((workSize + (threadPerBlock-1)) / threadPerBlock);

	std::cout << "Max block allowed: " << maxBlockOneIter << std::endl;
	std::cout << "Block Required: " << blocksRequired << std::endl;

	// return the min of either maxBlockOneIter or blocksRequired.
	return std::min(blocksRequired, maxBlockOneIter);
}
