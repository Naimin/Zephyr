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
		struct TriNode
		{
			TriNode() {}
			TriNode(const int v0, const int v1, const int v2)
			{
				verticesId[0] = v0;
				verticesId[1] = v1;
				verticesId[2] = v2;
			}

			Common::Vector3f normal;
			
			int verticesId[3];
		};

		struct TriEdge
		{
			TriEdge() { }

			int verticesId[2];
		};

		// specialization of iDualGraph
		class ZEPHYR_ALGORITHM_API TriDualGraph : public DualGraph<TriNode, TriEdge>
		{
		public:
			TriDualGraph();
			virtual ~TriDualGraph();

			void build(const Graphics::Mesh& mesh);

		private:
			std::vector<Vertex> mVertices;
		};
	}
}

#endif
