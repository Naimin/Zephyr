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
			TriNode() : Triangle() {}
		};

		struct TriEdge : public Common::Line
		{
			TriEdge() : Line() {}
		};

		// specialization of iDualGraph
		class TriDualGraph : DualGraph<TriNode, TriEdge>
		{
		public:
			TriDualGraph();
			virtual ~TriDualGraph();
		};
	}
}

#endif
