#ifndef TRI_DUAL_GRAPH_H
#define TRI_DUAL_GRAPH_H

#include "iDualGraph.h"
#include "Triangle.h"
#include <Line.h>
#include <Model.h>

namespace Zephyr
{
	namespace Algorithm
	{

		struct TriEdge;
		struct TriNode : public Common::Triangle
		{
			TriNode() {}
			TriNode(const Common::Vertex v0, const Common::Vertex v1, const Common::Vertex v2) : Triangle(v0,v1,v2) {}

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
			TriDualGraph(Common::Mesh* mesh);
			virtual ~TriDualGraph();

			void build(const Common::Mesh& mesh);

			// return the face id of the in each inStroke segment after passing the user inStrokes (multiple)
			Common::Model segment(const std::vector<std::vector<int>>& inStrokes);

		private:
			Common::Mesh* mpMesh;
		};
	}
}

#endif
