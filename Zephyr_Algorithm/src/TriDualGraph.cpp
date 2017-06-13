#include "TriDualGraph.h"
#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include <algorithm>
#include <ModelLoader.h>
#include <LNormUtil.h>

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
	const auto& vertices = mesh.getVertices();

	std::multimap<EdgeIdPair, int> edgesToNodeMap;

	// need to track Max   ||Ni - Nj||
	//                Edge            Inf
	float MaxE = 0;

	// create all the Node/Face
	for (int faceId = 0; faceId < mesh.getFaceCount(); ++faceId)
	{
		int currentId = faceId * 3;
		std::vector<int> index(3);
		index[0] = indices[currentId];
		index[1] = indices[currentId+1];
		index[2] = indices[currentId+2];

		auto vertex = vertices[index[0]].pos;
		auto point0 = Point(vertex.x, vertex.y, vertex.z);
		vertex = vertices[index[1]].pos;
		auto point1 = Point(vertex.x, vertex.y, vertex.z);
		vertex = vertices[index[2]].pos;
		auto point2 = Point(vertex.x, vertex.y, vertex.z);

		// add the node to the dual graph
		TriNode triNode(point0, point1, point2);
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
				int edgeId = linkNodes(relatedNodeId, currentNodeId);

				if (-1 == edgeId)
					continue;

				auto& edge = getEdge(edgeId);

				auto& node1 = getNode(currentNodeId);
				auto normal1 = node1.data.computeNormalNorm();
				auto& node2 = getNode(relatedNodeId);
				auto normal2 = node2.data.computeNormalNorm();

				//                 2
				// -B || Ni - Nj ||
				//                 Inf
				const float B = 1.0f;
				// Ni - Nj
				auto normalDiff = normal1 - normal2;
				// 
				float infNorm = Common::LNormUtil::LInfinityNorm(normalDiff);
				float infNormSq = infNorm * infNorm;
				float upperTerm = (-1.0f * B * infNormSq); 

				// need to track Max Edge LInfinityNorm
				MaxE = MaxE < infNorm ? infNorm : MaxE;

				// store the upper term first, later after all the edges is done, we get the lower term (depend on MaxE)
				edge.data.weight = upperTerm;
			}
			edgesToNodeMap.insert(std::make_pair(edgePair, currentNodeId));
		}
	}

	// for each edge, compute the actual weight
	// Actual weight = Exp ( UpperTerm / LowerTerm )
	// we already have UpperTerm computed above for each edge.data.weight
	float lowerTerm = MaxE;
	tbb::parallel_for_each(mEdges.begin(), mEdges.end(), [&](Edge& edge)
	{
		edge.data.weight = std::exp(edge.data.weight / lowerTerm);
	});
}

std::vector<std::vector<int>> Zephyr::Algorithm::TriDualGraph::segment(const std::vector<std::vector<int>>& inStrokes)
{
	// Data Term
	// Build X
	int numOfNode = (int)mNodes.size(); // n
	int numOfStrokes = (int)inStrokes.size(); // K

	// X = n X K // label matrix
	std::vector<std::vector<float>> X(numOfNode);
	tbb::parallel_for(0, numOfNode, [&](const int i)
	{
		X[i].resize(numOfStrokes);
	});

	// for each stroke, find the corresponding Xi and mark it
	tbb::parallel_for(0, (int)inStrokes.size(), [&](const int strokeId)
	{
		auto& stroke = inStrokes[strokeId];
		tbb::parallel_for(0, (int)stroke.size(), [&](const int nodeId)
		{
			X[nodeId][strokeId] = 1.0f; // mark the label vector
		});
	});

	// Propogation Term
	// build L
	// L = D - W

	// D
	std::vector<std::vector<float>> D(numOfNode);
	tbb::parallel_for(0, numOfNode, [&](const int i)
	{
		D[i].resize(numOfNode);
		
		// summation of adjacency weight
		auto pEdges = getNeighbourEdges(i);

		float weight = 0;
		for (auto edge : pEdges)
		{
			weight += edge.data.weight;
		}

		// diagonal matrix
		D[i][i] = weight;
	});
	
	// W
	std::vector<std::vector<float>> W(numOfNode);
	tbb::parallel_for(0, numOfNode, [&](const int i)
	{
		W[i].resize(numOfNode);
	});

	tbb::parallel_for(0, (int)mEdges.size(), [&](const int edgeId)
	{
		auto edge = getEdge(edgeId);

		auto f1 = edge.nodeIds[0];
		auto f2 = edge.nodeIds[1];

		// W   = w    if (fi, fj) in E
		//  ij    ij
		W[f1][f2] = edge.data.weight;
		// symmetry of the edge
		W[f2][f1] = edge.data.weight;
	});

	// L
	// L = D - W
	std::vector<std::vector<float>> L(numOfNode);
	tbb::parallel_for(0, numOfNode, [&](const int i)
	{
		L[i].resize(numOfNode);

		tbb::parallel_for(tbb::blocked_range2d<int, int>(0, numOfNode, 0, numOfNode), [&](const tbb::blocked_range2d<int, int>& r)
		{
			for (int x = r.rows().begin(); x != r.rows().end(); ++x)
			{
				for (int y = r.cols().begin(); y != r.cols().end(); ++y)
				{
					L[x][y] = D[x][y] - W[x][y];
				}
			}
		});
	});

	// Gradient Term
	// Build S

	// Formulate to solver



	return std::vector<std::vector<int>>();
}
