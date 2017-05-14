#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Graphics
	{
		class ShaderManager
		{
			public:
				enum SHADER_TYPE
				{
					VERTEX = 0,
					PIXEL
				};

				ShaderManager();
				virtual ~ShaderManager();

				D3D12_SHADER_BYTECODE getOrCreateShader(const std::wstring& shaderPath, const std::wstring& shaderName, SHADER_TYPE type);
				bool createShader(const std::wstring& shaderPath, const std::wstring& shaderName, SHADER_TYPE type);
				D3D12_SHADER_BYTECODE getShader(const std::wstring& shaderName);

			private:
				std::unordered_map<std::wstring, D3D12_SHADER_BYTECODE> mShaderByteCode;
		};
	}
}

#endif
