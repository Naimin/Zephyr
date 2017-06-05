#ifndef TRI_DUAL_GRAPH_H
#define TRI_DUAL_GRAPH_H

#include "iDualGraph.h"
#include <Triangle.h>
#include <Line.h>

namespace Zephyr
{
	namespace Algorithm
	{

		struct TriEdge;
		struct TriNode : public Common::Triangle
		{
			std::vector<TriEdge> edges;
		};

		struct TriEdge : public Common::Line
		{
			std::vector<TriNode> nodes;
		};

		// specialization of iDualGraph
		class TriDualGraph : DualGraph<TriNode, TriEdge>
		{
		public:
			TriDualGraph();
			virtual ~TriDualGraph();

			std::vector<Node> mNodes;
			std::vector<Edge> mEdges;
		};
	}
}

#endif
