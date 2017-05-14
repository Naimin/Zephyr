#pragma once

#ifdef ZEPHYR_GRAPHICS_EXPORTS  
#define ZEPHYR_GRAPHICS_API __declspec(dllexport)   
#else  
#define ZEPHYR_GRAPHICS_API __declspec(dllimport)   
#endif 

#include <memory>
#include <functional>
#include <map>
#include <vector>
#include <unordered_map>
#include <iostream>

// custom shared_ptr for directX resource
#include "SharedPtr.h"

#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include "d3dx12.h"

#include <d3d12sdklayers.h>