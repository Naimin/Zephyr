#ifndef PIPELINE_H
#define PIPELINE_H

#include "stdfx.h"

namespace Zephyr
{
	namespace Graphics
	{
		class GraphicsEngine;

		struct PipelineOption
		{
			D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlag;
			std::wstring vertexShader;
			std::wstring vertexShaderPath;
			std::wstring pixelShader;
			std::wstring pixelShaderPath;
			std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayout;
			int frameBufferCount;
			int constantBufferSize;
		};

		class Pipeline
		{
			public:
				Pipeline(GraphicsEngine* pEngine);
				virtual ~Pipeline();

				virtual bool initialize(const PipelineOption& option);

			public:
				ID3D12PipelineState* getPipeline() const;
				ID3D12RootSignature* getRootSignature() const;
				ID3D12DescriptorHeap* getDepthStencilDescriptorHeap() const;
				UINT8* getConstantBufferGPUAddress(const int frameIndex) const;
				ID3D12Resource* getConstantBufferUploadHeap(const int frameIndex) const;
				ID3D12DescriptorHeap* getConstantBufferDescriptorHeap(const int frameIndex) const;

			protected:
				virtual bool createRootSignature(const D3D12_ROOT_SIGNATURE_FLAGS& flag);
				virtual bool createConstantBuffer(const int frameBufferCount, const int constantBufferSize);
				virtual bool createInputLayout(const std::vector<D3D12_INPUT_ELEMENT_DESC>& inputLayout);
				virtual bool createShaders(const std::wstring& vertexShader, const std::wstring& vertexShaderPath, const std::wstring& pixelShader, const std::wstring& pixelShaderPath);
				virtual bool createDepthStencil();
				virtual bool createSampler();
				virtual bool createPipeline();

			protected:
				GraphicsEngine* mpEngine;
				ID3D12PipelineState* mpPipelineState; // pso containing a pipeline state

				std::vector<D3D12_INPUT_ELEMENT_DESC> mInputElements;
				D3D12_INPUT_LAYOUT_DESC mInputLayout;
				ID3D12RootSignature* mpRootSignature;
				ID3D12DescriptorHeap* mpDSDescriptorHeap; // This is a heap for our depth/stencil buffer descriptor
				std::vector<D3D12_SHADER_BYTECODE> mShaderByteCodes;

				std::vector<ID3D12DescriptorHeap*> mConstantBufferDescriptorHeap;
				std::vector<ID3D12Resource*> mConstantBufferUploadHeap;
				std::vector<UINT8*> mConstantBufferGPUAddress;

				std::vector<D3D12_STATIC_SAMPLER_DESC> mSamplerDesc;
		};
	}
}

#endif