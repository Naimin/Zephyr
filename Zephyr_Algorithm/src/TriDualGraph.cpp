#include "TriDualGraph.h"
#include <iostream>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <ModelLoader.h>

using namespace Zephyr::Common;
using namespace Zephyr::Graphics;

Zephyr::Algorithm::TriDualGraph::TriDualGraph()
{
	ModelLoader loader;
	auto filePath = "..\\model\\Lightning\\lightning_obj.obj";
	std::shared_ptr<RenderableModel> mpModel(new RenderableModel(L"Lightning"));
	mpModel->loadFromFile(filePath);

	auto mesh = mpModel->getMesh(0);

	build(mesh);

	auto result = getNeighbourNodeId(1);

	/*
	Triangle tri(Point(1.0f,1.0f,1.0f), Point(2.0f,2.0f,2.0f), Point(0.0f,0.0f,0.0f));
	auto area = tri.computeArea();

	std::cout << area;

	TriNode triNode1, triNode2;

	auto nodeId1 = addNode(triNode1);
	auto nodeId2 = addNode(triNode2);

	auto node = getNode(nodeId1);

	getNeighbourNodeId(nodeId1);

	linkNodes(nodeId1, nodeId2);
	*/
}

Zephyr::Algorithm::TriDualGraph::~TriDualGraph()
{

}


void Zephyr::Algorithm::TriDualGraph::build(const Graphics::Mesh & mesh)
{
	const auto& indices = mesh.getIndices();
	mVertices = mesh.getVertices();

	std::multimap<EdgeIdPair, int> edgesToNodeMap;

	// create all the Node/Face
	for (int faceId = 0; faceId < mesh.getFaceCount(); ++faceId)
	{
		int currentId = faceId * 3;
		std::vector<int> index(3);
		index[0] = indices[currentId];
		index[1] = indices[currentId+1];
		index[2] = indices[currentId+2];

		// add the node to the dual graph
		TriNode triNode(indices[0], indices[1], indices[2]);
		int currentNodeId = addNode(triNode);

		// sort the indices in order, so edge id pair will match
		std::sort(index.begin(), index.end());

		// create the edges and link with other node
		for (int i = 0; i < 3; ++i)
		{
			EdgeIdPair edgePair;
			edgePair = ((i != 2) ? EdgeIdPair(index[i], index[i + 1]) : EdgeIdPair(index[0], index[2]));

			// find all the node that share this edge and link them
			auto relatedNodeIds = edgesToNodeMap.equal_range(edgePair);
			for (auto relatedNodeItr = relatedNodeIds.first; relatedNodeItr != relatedNodeIds.second; ++relatedNodeItr)
			{
				int relatedNodeId = relatedNodeItr->second;
				// link the two node
				int edgeId = linkNodes(currentNodeId, relatedNodeId);
			}
			edgesToNodeMap.insert(std::make_pair(edgePair, currentNodeId));
		}
	}
}
