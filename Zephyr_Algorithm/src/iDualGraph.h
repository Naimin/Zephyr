#ifndef I_DUAL_GRAPH_H
#define I_DUAL_GRAPH_H

namespace Zephyr
{
	namespace Algorithm
	{
		template <class NodeType, class EdgeType> class DualGraph
		{
		public:
			typedef NodeType Node;
			typedef EdgeType Edge;

			DualGraph() {}

		protected:
			//Node* mpNodes;
			//Edge* mpEdges;
		};
	}
}

#endif