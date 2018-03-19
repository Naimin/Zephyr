#pragma once

#ifdef ZEPHYR_GPU_EXPORTS
#define ZEPHYR_GPU_API __declspec(dllexport)   
#else  
#define ZEPHYR_GPU_API __declspec(dllimport)   
#endif 

#ifndef NOMINMAX
#define NOMINMAX
#endif