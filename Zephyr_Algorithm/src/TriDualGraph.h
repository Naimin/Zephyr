#ifndef TRI_DUAL_GRAPH_H
#define TRI_DUAL_GRAPH_H

#include "iDualGraph.h"
#include "Triangle.h"
#include <Line.h>
#include <Mesh.h>

namespace Zephyr
{
	namespace Algorithm
	{

		struct TriEdge;
		struct TriNode : public Common::Triangle
		{
			TriNode() {}
			TriNode(const Common::Point v0, const Common::Point v1, const Common::Point v2) : Triangle(v0,v1,v2) {}

			int label;
		};

		struct TriEdge
		{
			TriEdge() : weight(-1.0f) { }

			float weight; 
		};

		// specialization of iDualGraph
		class ZEPHYR_ALGORITHM_API TriDualGraph : public DualGraph<TriNode, TriEdge>
		{
		public:
			TriDualGraph();
			virtual ~TriDualGraph();

			void build(const Common::Mesh& mesh);

			// return the face id of the in each inStroke segment after passing the user inStrokes (multiple)
			void segment(const std::vector<std::vector<int>>& inStrokes);

		};
	}
}

#endif
