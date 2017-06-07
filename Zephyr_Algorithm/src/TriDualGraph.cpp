#include "TriDualGraph.h"
#include <iostream>

using namespace Zephyr::Common;

Zephyr::Algorithm::TriDualGraph::TriDualGraph()
{
	Triangle tri(Point(1.0f,1.0f,1.0f), Point(2.0f,2.0f,2.0f), Point(0.0f,0.0f,0.0f));
	auto area = tri.computeArea();

	std::cout << area;

	TriNode triNode1, triNode2;

	auto nodeId1 = addNode(triNode1);
	auto nodeId2 = addNode(triNode2);

	auto node = getNode(nodeId1);

	getNeighbourNodeId(nodeId1);

	linkNodes(nodeId1, nodeId2);
}

Zephyr::Algorithm::TriDualGraph::~TriDualGraph()
{

}
