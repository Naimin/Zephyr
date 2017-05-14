#pragma once

#ifdef ZEPHYR_ALGORITHM_EXPORTS  
#define ZEPHYR_ALGORITHM_API __declspec(dllexport)   
#else  
#define ZEPHYR_ALGORITHM_API __declspec(dllimport)   
#endif 