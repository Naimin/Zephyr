#include "TriDualGraph.h"
#include <iostream>

using namespace Zephyr::Common;

Zephyr::Algorithm::TriDualGraph::TriDualGraph()
{
	Triangle tri(Point(1.0f,1.0f,1.0f), Point(2.0f,2.0f,2.0f), Point(0.0f,0.0f,0.0f));
	auto area = tri.computeArea();

	std::cout << area;

	

}

Zephyr::Algorithm::TriDualGraph::~TriDualGraph()
{

}
