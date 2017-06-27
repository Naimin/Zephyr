#include "TriDualGraph.h"
#include <iostream>
#include <sstream>
#include <set>

#include <tbb/parallel_for.h>

#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include <tbb/concurrent_vector.h>
#include <Eigen/SparseCholesky>
#include <Eigen/CholmodSupport>
#include <Eigen/SparseLU>
#include <algorithm>
#include <LNormUtil.h>
#include <ctime>

#include <MeshLoader.h>
#include <MeshExporter.h>

using namespace Zephyr::Common;
//using namespace Zephyr::Graphics;

#define DENSE 1

#ifdef DENSE
	typedef Eigen::MatrixXd matrix;
#else
	typedef Eigen::SparseMatrix<double> matrix;
#endif
	typedef Eigen::Triplet<double> triplet;

Zephyr::Algorithm::TriDualGraph::TriDualGraph(Common::Mesh* pMesh) : mpMesh(pMesh)
{
	//ModelLoader loader;
	//auto filePath = "..\\model\\Lightning\\lightning_obj.obj";
	//auto filePath = "..\\model\\bunny.obj";
	//std::shared_ptr<RenderableModel> mpModel(new RenderableModel(L"bunny"));
	//mpModel->loadFromFile(filePath);

	//Common::Model model;
	//Common::MeshLoader::loadFile(filePath, &model);

	//auto mesh = model.getMesh(0);

	if (nullptr == mpMesh)
	{
		std::cout << "Failed to construct TriDualGraph : no mesh supplied" << std::endl;
		return;
	}

	build(*pMesh);

	//Common::Model resultModel;

	//MeshExporter::exportMesh("D:\\sandbox\\bunny_result.obj", &model);
}

Zephyr::Algorithm::TriDualGraph::~TriDualGraph()
{

}


void Zephyr::Algorithm::TriDualGraph::build(const Common::Mesh & mesh)
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

		auto point0 = vertices[index[0]];
		auto point1 = vertices[index[1]];
		auto point2 = vertices[index[2]];

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

Zephyr::Common::Model Zephyr::Algorithm::TriDualGraph::segment(const std::vector<std::vector<int>>& inStrokes)
{
	// Data Term
	// Build X
	int numOfNode = (int)mNodes.size(); // n
	int numOfStrokes = (int)inStrokes.size(); // K

	matrix A(numOfNode, numOfNode);

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
	// accumulate the result
	A = Ip;

	// clear Ip
	Ip.resize(0, 0);


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

	// acummulate into A
	A += L * L; // get L squared

	// Clear D, W, L
	D.resize(0, 0);
	W.resize(0, 0);
	L.resize(0, 0);

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

	A += 2 * (S * S);
	
	S.resize(0, 0);

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

	// clear Xc
	Xc.resize(0, 0);

	// Formulate to solver
	// solve (Ip + L^2 + 2S^2)X = B
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

	// put the segmentation result into segments vector
	std::vector<tbb::concurrent_vector<int>> segments(numOfStrokes);
	for (int k = 0; k < numOfStrokes; ++k)
	{
		//std::cout << segmentCounters[k] << std::endl;
		segments[k].reserve(segmentCounters[k]);
	}

	tbb::parallel_for(0, numOfNode, [&](const int i)
	{
		auto label = mNodes[i].data.label;

		segments[label].push_back(i);
	});

	// Debug : print out the segment size
	for (int k = 0; k < numOfStrokes; ++k)
	{
		std::cout << segments[k].size() << std::endl;
	}

	// Construct the final result model
	// each label is a mesh

	Model resultModel;
	resultModel.resizeMeshes(numOfStrokes * 2); // give the user selected face a different color

	for (int label = 0; label < numOfStrokes; ++label)
	{
		const Vector3f labelColor = Vector3f((float)std::rand() / RAND_MAX, (float)std::rand() / RAND_MAX, (float)std::rand() / RAND_MAX);
		Material material;
		material.mDiffuseColor = labelColor * 0.5;

		std::stringstream ss;
		ss << label << "_mat";
		material.mName = ss.str();
		int currentMaterialId = resultModel.getMaterialsCount();
		resultModel.addMaterial(material);

		auto& mesh = resultModel.getMesh(label);
		mesh.setMaterialId(currentMaterialId);

		// setup the selected mesh
		Material selectedMaterial;
		std::stringstream ss2;
		ss2 << "selected_" << label << "_mat";
		selectedMaterial.mName = ss2.str();
		selectedMaterial.mDiffuseColor = labelColor;
		currentMaterialId = resultModel.getMaterialsCount();
		resultModel.addMaterial(selectedMaterial);
		auto& selectedMesh = resultModel.getMesh(numOfStrokes + label);
		selectedMesh.setMaterialId(currentMaterialId);

		auto userStrokes = inStrokes[label];
		std::set<int> strokeSet; // put in a set to speed up search
		strokeSet.insert(userStrokes.begin(), userStrokes.end());

		int userSelectedFaceCount = (int)userStrokes.size();
		selectedMesh.resizeIndices(userSelectedFaceCount * 3);
		auto& selectedIndices = selectedMesh.getIndices();

		auto& segment = segments[label];

		std::map<Vertex, int> vertexIdMap;
		std::map<Vertex, int> selectedVertexIdMap;

		// resize the index buffer
		mesh.resizeIndices(((int)segment.size() - userSelectedFaceCount) * 3);

		auto& indices = mesh.getIndices();

		int index = 0;
		int selectedIndex = 0;
		for (int i = 0; i < (int)segment.size(); ++i)
		{
			auto id = segment[i];
			auto& node = mNodes[id];

			// check if this face is user selected or not
			bool bIsSelected = false;
			if (strokeSet.find(id) != strokeSet.end())
				bIsSelected = true;

			int& localIndex = bIsSelected ? selectedIndex : index;
			auto& localIndices = bIsSelected ? selectedIndices : indices;
			auto& localVertexIdMap = bIsSelected ? selectedVertexIdMap : vertexIdMap;
			for (int k = 0; k < 3; ++k)
			{
				Vertex& v = node.data.mVertex[k];
				// set the label vertex color
				//v.color = labelColor;

				// check if this vertex already exist
				auto itr = localVertexIdMap.find(v);
				if (itr == localVertexIdMap.end())
					localVertexIdMap.insert(std::make_pair(v,(int)localVertexIdMap.size()));

				localIndices[localIndex * 3 + k] = localVertexIdMap[v];
			}
			++localIndex;
		}

		// insert the vertex buffer
		mesh.resizeVertices((int)vertexIdMap.size());
		auto& vertices = mesh.getVertices();
		tbb::parallel_for_each(vertexIdMap.begin(), vertexIdMap.end(), [&](const std::pair<Vertex, int>& itr)
		{
			vertices[itr.second] = itr.first;
		});

		selectedMesh.resizeVertices((int)selectedVertexIdMap.size());
		auto& selectedVertices = selectedMesh.getVertices();
		tbb::parallel_for_each(selectedVertexIdMap.begin(), selectedVertexIdMap.end(), [&](const std::pair<Vertex, int>& itr)
		{
			selectedVertices[itr.second] = itr.first;
		});
	}

	boost::filesystem::path outputPath("D:\\sandbox\\");
	std::stringstream ss;
	ss << "result" << "_bunny" << ".obj";
	outputPath /= ss.str();
	Common::MeshExporter::exportMesh(outputPath.string(), &resultModel);

	return resultModel;
}
