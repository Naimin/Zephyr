#include "TriDualGraph.h"
#include <iostream>

#include <tbb/parallel_for.h>

#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include <tbb/concurrent_vector.h>
#include <Eigen/SparseCholesky>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseLU>
#include <algorithm>
#include <ModelLoader.h>
#include <LNormUtil.h>
#include <ctime>

using namespace Zephyr::Common;
using namespace Zephyr::Graphics;

#define DENSE 1

#ifdef DENSE
	typedef Eigen::MatrixXd matrix;
#else
	typedef Eigen::SparseMatrix<double> matrix;
#endif
	typedef Eigen::Triplet<double> triplet;

Zephyr::Algorithm::TriDualGraph::TriDualGraph()
{
	ModelLoader loader;
	//auto filePath = "..\\model\\Lightning\\lightning_obj.obj";
	auto filePath = "..\\model\\bunny.obj";
	std::shared_ptr<RenderableModel> mpModel(new RenderableModel(L"bunny"));
	mpModel->loadFromFile(filePath);

	auto mesh = mpModel->getMesh(0);

	build(mesh);

	auto result = getNeighbourNodeId(1);

	std::vector<std::vector<int>> input;
	input.push_back(std::vector<int>());
	input.back().push_back(5);
	/*input.back().push_back(2);
	input.back().push_back(3);
	input.back().push_back(4);*/

	input.push_back(std::vector<int>());
	input.back().push_back(7);
	/*input.back().push_back(101);
	input.back().push_back(102);
	input.back().push_back(103);*/

	input.push_back(std::vector<int>());
	input.back().push_back(10);
	/*input.back().push_back(2001);
	input.back().push_back(2002);
	input.back().push_back(2003);*/

	input.push_back(std::vector<int>());
	input.back().push_back(15);
	/*input.back().push_back(3001);
	input.back().push_back(3002);
	input.back().push_back(3003);*/

	segment(input);

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

void Zephyr::Algorithm::TriDualGraph::segment(const std::vector<std::vector<int>>& inStrokes)
{
	// Data Term
	// Build X
	int numOfNode = (int)mNodes.size(); // n
	int numOfStrokes = (int)inStrokes.size(); // K

	// X = n X K // label matrix
	matrix X(numOfNode, numOfStrokes);
	X.setZero();
	tbb::concurrent_vector<triplet> XCoef;
	// Ip =  Ip(fi,fi)  =  1 if fi is marked by any label.
	matrix Ip(numOfNode, numOfNode);
	Ip.setZero();
	tbb::concurrent_vector<triplet> IpCoef;
	// Xc = label vector
	matrix Xc(numOfStrokes, numOfStrokes);
	Xc.setIdentity();
	tbb::concurrent_vector<triplet> XcCoef;

	// for each stroke, find the corresponding Xi and mark it
	//tbb::serial::parallel_for(0, (int)inStrokes.size(), [&](const int strokeId)
	for (int strokeId = 0; strokeId < (int)inStrokes.size(); ++strokeId)
	{
		auto& stroke = inStrokes[strokeId];
		//tbb::serial::parallel_for(0, (int)stroke.size(), [&](const int nodeId)
		for (int nodeId = 0; nodeId < (int)stroke.size(); ++nodeId)
		{
#ifdef DENSE
			X(nodeId, strokeId) = 1.0;
			Ip(nodeId, nodeId) = 1.0;
#else
			XCoef.push_back(triplet(nodeId, strokeId, 1.0)); // mark the label vector
			IpCoef.push_back(triplet(nodeId, nodeId, 1.0)); // set the Ip matrix
#endif	
		}//);
#ifdef DENSE
		//Xc(strokeId, strokeId) = 1.0;
#else
		XcCoef.push_back(triplet(strokeId, strokeId, 1.0));
#endif
	}//);
#ifndef DENSE
	X.setFromTriplets(XCoef.begin(), XCoef.end());
	XCoef.clear();
	Ip.setFromTriplets(IpCoef.begin(), IpCoef.end());
	IpCoef.clear();
	Xc.setFromTriplets(XcCoef.begin(), XcCoef.end());
	XcCoef.clear();
#endif
	//std::cout << Xc << std::endl;

	// Propogation Term
	// build L
	// L = D - W

	// D
	matrix D(numOfNode, numOfNode);
	D.setZero();
	tbb::concurrent_vector<triplet> DCoef;
	//tbb::serial::parallel_for(0, numOfNode, [&](const int i)
	for (int i = 0; i < numOfNode; ++i)
	{
		// summation of adjacency weight
		auto pEdges = getNeighbourEdges(i);

		float weight = 0;
		for (auto edge : pEdges)
		{
			weight += edge.data.weight;
		}

		// diagonal matrix
#ifdef DENSE	
		D(i, i) = weight;
#else		
		DCoef.push_back(triplet(i, i, weight));
#endif
	}//);
#ifndef DENSE
	D.setFromTriplets(DCoef.begin(), DCoef.end());
	DCoef.clear();
#endif
	//std::cout << D;

	// W, weighted adjacency matrix
	matrix W(numOfNode, numOfNode);
	W.setZero();
	tbb::concurrent_vector<triplet> WCoef;
	//tbb::serial::parallel_for(0, (int)mEdges.size(), [&](const int edgeId)
	for (int edgeId = 0; edgeId < (int)mEdges.size(); ++edgeId)
	{
		auto edge = getEdge(edgeId);

		auto f1 = edge.nodeIds[0];
		auto f2 = edge.nodeIds[1];

		// W   = w    if (fi, fj) in E
		//  ij    ij
#ifdef DENSE
		W(f2, f1) = edge.data.weight;
		W(f1, f2) = edge.data.weight;
#else
		WCoef.push_back(triplet(f1, f2, edge.data.weight));
		// symmetry of the edge
		WCoef.push_back(triplet(f2, f1, edge.data.weight));
#endif
	}//);
#ifndef DENSE
	W.setFromTriplets(WCoef.begin(), WCoef.end());
	WCoef.clear();
#endif

	// L
	// L = D - W
	int c = 0;
	matrix L(numOfNode, numOfNode);
	L.setZero();
	tbb::concurrent_vector<triplet> LCoef;
	tbb::blocked_range2d<int, int> r(0, numOfNode, 0, numOfNode);
	//tbb::serial::parallel_for(tbb::blocked_range2d<int, int>(0, numOfNode, 0, numOfNode), [&](const tbb::blocked_range2d<int, int>& r)
	{
		for (int x = r.rows().begin(); x != r.rows().end(); ++x)
		{
			for (int y = r.cols().begin(); y != r.cols().end(); ++y)
			{
#ifdef DENSE
				L(x, y) = D(x, y) - W(x, y);
#else
				LCoef.push_back(triplet(x, y, D.coeff(x, y) - W.coeff(x, y)));
#endif
			}
		}
	}//);
#ifndef DENSE
	L.setFromTriplets(LCoef.begin(), LCoef.end());
	LCoef.clear();
#endif

	// Gradient Term
	// Build S
	matrix S(numOfNode, numOfNode);
	tbb::concurrent_vector<triplet> SCoef;
	S.setIdentity(); // make identity

	// mark each edge as -1.0f
	//tbb::serial::parallel_for(0, (int)mEdges.size(), [&](const int i)
	for (int i = 0; i < (int)mEdges.size(); ++i)
	{
		auto& edge = getEdge(i);

		auto f1 = edge.nodeIds[0];
		auto f2 = edge.nodeIds[1];

#ifdef DENSE
		S(f1, f2) = -1.0;
		S(f2, f1) = -1.0;
#else
		SCoef.push_back(triplet(f1, f2, -1.0));
		SCoef.push_back(triplet(f2, f1, -1.0));
#endif
	}//);
#ifndef DENSE
	S.setFromTriplets(SCoef.begin(), SCoef.end());
	SCoef.clear();
#endif

	// Build B
	// For each face that is labeled assign Xc(strokeId)
	matrix B(numOfNode, numOfStrokes);
	B.setZero();
	tbb::concurrent_vector<triplet> BCoef;
	//tbb::serial::parallel_for(0, (int)inStrokes.size(), [&](const int strokeId)
	for (int strokeId = 0; strokeId < (int)inStrokes.size(); ++strokeId)
	{
		auto& stroke = inStrokes[strokeId];
		//tbb::serial::parallel_for(0, (int)stroke.size(), [&](const int nodeId)
		for (auto nodeId : stroke)
		{
			for (int k = 0; k < numOfStrokes; ++k)
			{
#ifdef DENSE
				B(nodeId, k) = Xc(strokeId, k);
#else
				BCoef.push_back(triplet(nodeId, k, Xc.coeff(strokeId, k)));
#endif			
			}
		}//);
	}//);
#ifndef DENSE
	B.setFromTriplets(BCoef.begin(), BCoef.end());
	BCoef.clear();
#endif

	// Formulate to solver
	// solve (Ip + L^2 + 2S^2)X = B
	auto LSq = L * L; // get L squared
	auto SSq = S * S;
	auto twoSSq = 2 * SSq;

	auto A = (Ip + LSq + twoSSq).eval();

	std::cout << "Size of A : " << A.rows() << " X " << A.cols() << std::endl;
	std::cout << "Size of X : " << X.rows() << " X " << X.cols() << std::endl;
	std::cout << "Size of B : " << B.rows() << " X " << B.cols() << std::endl;

	//std::cout << B << std::endl;

	std::cout << "Building Cholesky... ";
	auto time = std::clock();

#ifdef DENSE
	Eigen::SparseMatrix<double> sparse = A.sparseView();
#else
	Eigen::SparseMatrix<double> sparse = A;
#endif

	Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>> chol;
	chol.analyzePattern(sparse);
	chol.factorize(sparse);

	std::cout << "Done in " << (std::clock() - time) / CLOCKS_PER_SEC << " Secs" << std::endl;

	std::cout << "Solving... ";
	time = std::clock();

	auto solveX = chol.solve(B);
	std::cout << "Done in " << (std::clock() - time) / CLOCKS_PER_SEC << " Secs" << std::endl;

	// evaluate the final result
	auto eval = solveX.eval();

	std::cout << eval.rows() << " X " << eval.cols() << std::endl;
	//std::cout << eval;

	std::cout << "Segmenting... ";
	time = std::clock();

	// Label the result
	std::vector<std::vector<int>> segment(numOfStrokes);

	std::vector<tbb::atomic<int>> segmentCounters(numOfStrokes);
	tbb::parallel_for(0, numOfNode, [&](const int i)
	//for (int i = 0; i < numOfNode; ++i)
	{
		double max = 0;
		int maxId = -1;

		for (int k = 0; k < numOfStrokes; ++k)
		{
#ifdef DENSE
			double value = std::abs(eval(i, k));
#else
			double value = std::abs(eval.coeff(i, k));
#endif
			maxId = value > max ? k : maxId;
			max = std::max(value, max);
		}

		mNodes[i].data.label = maxId;
		segmentCounters[maxId].fetch_and_increment();
	});
	std::cout << "Done in " << (std::clock() - time) / CLOCKS_PER_SEC << " Secs" << std::endl;

	for (int k = 0; k < numOfStrokes; ++k)
	{
		std::cout << segmentCounters[k] << std::endl;
	}
}
