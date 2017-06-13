#ifndef I_DUAL_GRAPH_H
#define I_DUAL_GRAPH_H

#include "stdfx.h"
#include <vector>
#include <algorithm>

namespace Zephyr
{
	namespace Algorithm
	{
		template <class NodeType, class EdgeType> class ZEPHYR_ALGORITHM_API DualGraph
		{
		public:

			struct Node
			{
				Node(const NodeType data_) : data(data_) {}

				NodeType data;
				std::vector<int> edgeIds;
			};

			struct Edge
			{
				Edge() {}
				Edge(const EdgeType data_) : data(data_) {}

				EdgeType data;
				std::vector<int> nodeIds;
			};

			DualGraph() {}
			~DualGraph() {}

			// access both node and edge by id
			Node& getNode(const int id) { return mNodes[id]; }
			Node getNode(const int id) const { return mNodes[id]; }

			Edge& getEdge(const int id) { return mEdges[id]; }
			Edge getEdge(const int id) const { return mEdges[id]; }

			// construction function
			int addNode(const NodeType& data)
			{
				mNodes.push_back(Node(data));
				return (int)mNodes.size() - 1; // return new node id
			}

			int addEdge() 
			{  
				mEdges.push_back(Edge());
				return (int)mEdges.size() - 1;
			}

			int addEdge(const EdgeType& data)
			{
				mEdges.push_back(Edge(data));
				return (int)mEdges.size() - 1; // return new edge id
			}

			int linkNodes(const int currentNodeId, const int linkNodeId)
			{
				if (currentNodeId == linkNodeId)
				{
					std::cout << "Attempting to link the same node to itself";
					return -1;
				}

				Node& currentNode = mNodes[currentNodeId];
				Node& linkNode = mNodes[linkNodeId];

				// create a new edge
				int edgeId = addEdge();
				Edge& edge = getEdge(edgeId);

				// if edge don't already exist in this node
				if (std::find(currentNode.edgeIds.begin(), currentNode.edgeIds.end(), edgeId) == currentNode.edgeIds.end())
					currentNode.edgeIds.push_back(edgeId);

				if (std::find(linkNode.edgeIds.begin(), linkNode.edgeIds.end(), edgeId) == linkNode.edgeIds.end())
					linkNode.edgeIds.push_back(edgeId);

				// since it is a new edge no need to check if edges exist.
				edge.nodeIds.push_back(currentNodeId);
				edge.nodeIds.push_back(linkNodeId);

				return edgeId;
			}

			// transverse utility
			std::vector<int> getNeighbourNodeId(const int id) const
			{
				const Node& currentNode = mNodes[id];

				std::vector<int> neighbour;
				for (auto edgeId : currentNode.edgeIds)
				{
					for (auto nodeId : mEdges[edgeId].nodeIds)
					{
						// if not current/self id
						if (nodeId != id)
							neighbour.push_back(nodeId);
					}
				}
				return neighbour;
			}

			std::vector<Edge> getNeighbourEdges(const int id) const
			{
				const Node& currentNode = mNodes[id];

				std::vector<Edge> neighbour;
				for (auto edgeId : currentNode.edgeIds)
				{
					neighbour.push_back(getEdge(edgeId));
				}
				return neighbour;
			}

		protected:
			std::vector<Node> mNodes;
			std::vector<Edge> mEdges;
		};
		typedef std::pair<int, int> EdgeIdPair;
	}
}

#endif