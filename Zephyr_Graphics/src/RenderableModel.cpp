#include "RenderableModel.h"
#include "ResourceManager.h"
#include <sstream>
#include "CommandList.h"
#include "Zephyr_Graphics.h"
#include "StringUtils.h"
#include <IO/MeshLoader.h>
#include <boost/filesystem/path.hpp>

using namespace Zephyr;
using namespace Zephyr::Common;

Graphics::RenderableModel::RenderableModel(const std::wstring & name, ResourceManager* pResourceManager) : mName(name), mpResourceManager(pResourceManager)
{
}

Graphics::RenderableModel::~RenderableModel()
{
	unloadFromGPU();
}

D3D12_VERTEX_BUFFER_VIEW Graphics::RenderableModel::getVertexResourceView(const int meshId)
{
	return mpResourceManager->getVertexResourceView(getVertexResourceName(meshId), sizeof(Common::Vertex));
}

D3D12_INDEX_BUFFER_VIEW Graphics::RenderableModel::getIndexResourceView(const int meshId)
{
	return mpResourceManager->getIndexResourceView(getIndexResourceName(meshId));
}

bool Graphics::RenderableModel::loadFromFile(const std::string & path)
{
	mPath = boost::filesystem::path(path).wstring();
	mName = this->mPath;
	return Common::MeshLoader::loadFile(path, this);
}

bool Graphics::RenderableModel::loadOpenMesh(const OpenMeshMesh& mesh)
{
	return false;
}

bool Graphics::RenderableModel::uploadToGPU()
{
	for (int i = 0; i < (int)mMeshes.size(); ++i)
	{
		auto success = uploadVerticesToGPU(i);
		if (!success)
			return false;

		success = uploadIndicesToGPU(i);
		if (!success)
			return false;
	}

	for (int i = 0; i < (int)mMaterials.size(); ++i)
	{
		auto success = uploadTextureToGPU(i);
		if (!success)
			return false;
	}

	return true;
}

void Graphics::RenderableModel::unloadFromGPU()
{
	if (nullptr == mpResourceManager)
		return;

	std::wstringstream vertexBufferName;
	vertexBufferName << mName << L"_VB";
	mpResourceManager->releaseResource(vertexBufferName.str());

	std::wstringstream indexBufferName;
	indexBufferName << mName << L"_IB";
	mpResourceManager->releaseResource(indexBufferName.str());

	for (auto descriptorHeap : mTextureHeap)
	{
		if(nullptr != descriptorHeap)
			descriptorHeap->Release();
	}
	mTextureHeap.clear();
}

bool Graphics::RenderableModel::drawMesh(const int meshId, SharedPtr<ID3D12GraphicsCommandList> pCommandList)
{
	if (meshId >= mMeshes.size())
		return false;

	auto& mesh = mMeshes[meshId];
	auto& textureDescriptorHeap = mTextureHeap[mesh.getMaterialId()];

	if (nullptr != textureDescriptorHeap)
	{
		// set the descriptor heap
		ID3D12DescriptorHeap* descriptorHeaps[] = { textureDescriptorHeap };
		pCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
		// set the descriptor table to the descriptor heap (parameter 1, as constant buffer root descriptor is parameter index 0)
		pCommandList->SetGraphicsRootDescriptorTable(1, textureDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
	}

	pCommandList->IASetVertexBuffers(0, 1, &getVertexResourceView(meshId)); // set the vertex buffer (using the vertex buffer view)
	pCommandList->IASetIndexBuffer(&getIndexResourceView(meshId));
	pCommandList->DrawIndexedInstanced((UINT)mesh.getIndices().size(), 1, 0, 0, 0); 

	return true;
}

std::wstring Graphics::RenderableModel::getVertexResourceName(const int meshId) const
{
	std::wstringstream vertexBufferName;
	vertexBufferName << mName << L"_VB_" << meshId;

	return std::wstring(vertexBufferName.str());
}

std::wstring Graphics::RenderableModel::getIndexResourceName(const int meshId) const
{
	std::wstringstream indexBufferName;
	indexBufferName << mName << L"_IB_" << meshId;

	return std::wstring(indexBufferName.str());
}

bool Graphics::RenderableModel::uploadVerticesToGPU(const int meshId)
{
	if (meshId >= mMeshes.size())
		return false;

	auto mesh = mMeshes[meshId];
	auto vertexResName = getVertexResourceName(meshId);
	int vertexBufferSize = (int)(mesh.getVerticesCount() * sizeof(Vertex));

	auto success = mpResourceManager->createAndCopyToGPU(vertexResName, vertexBufferSize, &mesh.getVertices()[0]);
	if (!success)
		return false;
	
	return true;
}

bool Graphics::RenderableModel::uploadIndicesToGPU(const int meshId)
{
	if (meshId >= mMeshes.size())
		return false;

	auto mesh = mMeshes[meshId];
	auto indexResName = getIndexResourceName(meshId);
	int indexBufferSize = (int)(mesh.getIndicesCount() * sizeof(int));

	auto success = mpResourceManager->createAndCopyToGPU(indexResName, indexBufferSize, &mesh.getIndices()[0]);
	if (!success)
		return false;

	return true;
}

bool Graphics::RenderableModel::uploadTextureToGPU(const int materialId)
{
	if (materialId >= mMaterials.size())
		return false;

	mTextureHeap.push_back(nullptr);

	// if null
	if (mMaterials[materialId].getPath().empty())
		return true;

	Texture texture = Texture(mMaterials[materialId].getPath());
	if (!texture.isValid()) // check if the texture load is successful
		return false;

	auto rawData = texture.getRawData();

	// now describe the texture with the information we have obtained from the image
	D3D12_RESOURCE_DESC mResourceDesc = {};
	mResourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	mResourceDesc.Alignment = 0; // may be 0, 4KB, 64KB, or 4MB. 0 will let runtime decide between 64KB and 4MB (4MB for multi-sampled textures)
	mResourceDesc.Width = rawData.getWidth(); // width of the texture
	mResourceDesc.Height = rawData.getHeight(); // height of the texture
	mResourceDesc.DepthOrArraySize = 1; // if 3d image, depth of 3d image. Otherwise an array of 1D or 2D textures (we only have one image, so we set 1)
	mResourceDesc.MipLevels = 1; // Number of mipmaps. We are not generating mipmaps for this texture, so we have only one level
	mResourceDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // This is the dxgi format of the image (format of the pixels) // free image use BGRA format
	mResourceDesc.SampleDesc.Count = 1; // This is the number of samples per pixel, we just want 1 sample
	mResourceDesc.SampleDesc.Quality = 0; // The quality level of the samples. Higher is better quality, but worse performance
	mResourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; // The arrangement of the pixels. Setting to unknown lets the driver choose the most efficient one
	mResourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE; // no flags

	auto pDevice = mpResourceManager->getEngine()->getRenderer()->getDevice();

	UINT64 textureUploadBufferSize;
	// this function gets the size an upload buffer needs to be to upload a texture to the gpu.
	// each row must be 256 byte aligned except for the last row, which can just be the size in bytes of the row
	// eg. textureUploadBufferSize = ((((width * numBytesPerPixel) + 255) & ~255) * (height - 1)) + (width * numBytesPerPixel);
	//textureUploadBufferSize = (((imageBytesPerRow + 255) & ~255) * (textureDesc.Height - 1)) + imageBytesPerRow;
	pDevice->GetCopyableFootprints(&mResourceDesc, 0, 1, 0, nullptr, nullptr, nullptr, &textureUploadBufferSize);

	auto wpath = mMaterials[materialId].getPath().wstring();
	auto success = mpResourceManager->createTextureAndCopyToGPU(wpath, &mResourceDesc, (int)textureUploadBufferSize, rawData.accessPixels(), rawData.getLine(), rawData.getHeight());
	if (!success)
		return false;

	if (FAILED(pDevice->GetDeviceRemovedReason()))
		return nullptr;

	D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
	heapDesc.NumDescriptors = 1;
	heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	auto hr = pDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mTextureHeap[materialId]));
	if (FAILED(hr))
		return false;

	if (FAILED(pDevice->GetDeviceRemovedReason()))
		return nullptr;

	// now we create a shader resource view (descriptor that points to the texture and describes it)
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // free image use BGRA format
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = 1;
	pDevice->CreateShaderResourceView(mpResourceManager->getResource(wpath).get(), &srvDesc, mTextureHeap[materialId]->GetCPUDescriptorHandleForHeapStart());

	if (FAILED(pDevice->GetDeviceRemovedReason()))
		return nullptr;

	return true;
}
