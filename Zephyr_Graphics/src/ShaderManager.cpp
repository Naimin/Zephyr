#include "ShaderManager.h"

Zephyr::Graphics::ShaderManager::ShaderManager()
{
}

Zephyr::Graphics::ShaderManager::~ShaderManager()
{
}

D3D12_SHADER_BYTECODE Zephyr::Graphics::ShaderManager::getOrCreateShader(const std::wstring & shaderPath, const std::wstring & shaderName, SHADER_TYPE type)
{
	auto itr = mShaderByteCode.find(shaderName);
	if (itr == mShaderByteCode.end()) // not found in cache, need to create the shader
		createShader(shaderPath, shaderName, type);

	return mShaderByteCode[shaderName];
}

bool Zephyr::Graphics::ShaderManager::createShader(const std::wstring & shaderPath, const std::wstring & shaderName, SHADER_TYPE type)
{
	std::string shaderVersion;
	switch (type)
	{
		case VERTEX: shaderVersion = "vs_5_0"; break;
		case PIXEL: shaderVersion = "ps_5_0"; break;
	}
	// compile vertex shader
	ID3DBlob* pShader; // d3d blob for holding vertex shader bytecode
	ID3DBlob* pErrorBuff; // a buffer holding the error data if any
	auto hr = D3DCompileFromFile(shaderPath.c_str(),
		nullptr,
		nullptr,
		"main",
		shaderVersion.c_str(),
		D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
		0,
		&pShader,
		&pErrorBuff);
	if (FAILED(hr))
	{
		std::cout << ((char*)pErrorBuff->GetBufferPointer()) << std::endl;
		return false;
	}

	D3D12_SHADER_BYTECODE bytecode = {};
	bytecode.BytecodeLength = pShader->GetBufferSize();
	bytecode.pShaderBytecode = pShader->GetBufferPointer();
	mShaderByteCode[shaderName] = bytecode;

	return true;
}

D3D12_SHADER_BYTECODE Zephyr::Graphics::ShaderManager::getShader(const std::wstring & shaderName)
{
	return mShaderByteCode[shaderName];
}
