#pragma once

#ifdef ZEPHYR_COMMON_EXPORTS
#define ZEPHYR_COMMON_API __declspec(dllexport)   
#else  
#define ZEPHYR_COMMON_API __declspec(dllimport)   
#endif 

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "GeometryMath.h"