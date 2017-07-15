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

typedef Eigen::SparseMatrix<double> matrix;
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
	//tbb::concurrent_vector<triplet> XCoef;
	std::vector<triplet> XCoef;

	// Ip =  Ip(fi,fi)  =  1 if fi is marked by any label.
	matrix Ip(numOfNode, numOfNode);
	//tbb::concurrent_vector<triplet> IpCoef;
	std::vector<triplet> IpCoef;

	// Xc = label vector
	matrix Xc(numOfStrokes, numOfStrokes);
	//tbb::concurrent_vector<triplet> XcCoef;
	std::vector<triplet> XcCoef;

	// for each stroke, find the corresponding Xi and mark it
	//tbb::serial::parallel_for(0, (int)inStrokes.size(), [&](const int strokeId)
	for (int strokeId = 0; strokeId < (int)inStrokes.size(); ++strokeId)
	{
		auto& stroke = inStrokes[strokeId];
		//tbb::serial::parallel_for(0, (int)stroke.size(), [&](const int nodeId)
		for (int nodeId = 0; nodeId < (int)stroke.size(); ++nodeId)
		{
			XCoef.push_back(triplet(nodeId, strokeId, 1.0)); // mark the label vector
			IpCoef.push_back(triplet(nodeId, nodeId, 1.0)); // set the Ip matrix
		}//);
		XcCoef.push_back(triplet(strokeId, strokeId, 1.0));
	}//);
	X.setFromTriplets(XCoef.begin(), XCoef.end());
	XCoef.clear();
	Ip.setFromTriplets(IpCoef.begin(), IpCoef.end());
	IpCoef.clear();
	Xc.setFromTriplets(XcCoef.begin(), XcCoef.end());
	XcCoef.clear();
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
	//tbb::concurrent_vector<triplet> DCoeff;
	std::vector<triplet> DCoeff;
	
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
		DCoeff.push_back(triplet(i, i, weight));
	}//);
	D.setFromTriplets(DCoeff.begin(), DCoeff.end());
	DCoeff.clear();
	//std::cout << D;

	// W, weighted adjacency matrix
	matrix W(numOfNode, numOfNode);
	//tbb::concurrent_vector<triplet> WCoef;
	std::vector<triplet> WCoeff;
	//tbb::serial::parallel_for(0, (int)mEdges.size(), [&](const int edgeId)
	for (int edgeId = 0; edgeId < (int)mEdges.size(); ++edgeId)
	{
		auto edge = getEdge(edgeId);

		auto f1 = edge.nodeIds[0];
		auto f2 = edge.nodeIds[1];

		// W   = w    if (fi, fj) in E
		//  ij    ij
		WCoeff.push_back(triplet(f1, f2, edge.data.weight));
		// symmetry of the edge
		WCoeff.push_back(triplet(f2, f1, edge.data.weight));
	}//);
	W.setFromTriplets(WCoeff.begin(), WCoeff.end());
	WCoeff.clear();

	// L
	// L = D - W
	int c = 0;
	matrix L(numOfNode, numOfNode);
	//tbb::concurrent_vector<triplet> LCoef;
	std::vector<triplet> LCoeff;
	std::set<std::pair<int, int>> nonZeroIndex;

	// find non-zero in D
	for (int k = 0; k < D.outerSize(); ++k)
	{
		for (matrix::InnerIterator it(D, k); it; ++it)
		{
			nonZeroIndex.insert(std::make_pair(it.row(), it.col()));
		}
	}

	// find non-zero in W
	for (int k = 0; k < W.outerSize(); ++k)
	{
		for (matrix::InnerIterator it(W, k); it; ++it)
		{
			nonZeroIndex.insert(std::make_pair(it.row(), it.col()));
		}
	}

	for (auto i : nonZeroIndex)
	{
		LCoeff.push_back(triplet(i.first, i.second, D.coeff(i.first, i.second) - W.coeff(i.first, i.second)));
	}

	L.setFromTriplets(LCoeff.begin(), LCoeff.end());
	LCoeff.clear();

	// Clear D and W
	D.resize(0, 0);
	W.resize(0, 0);

	// acummulate into A
	A += L * L; // get L squared

	// Clear L
	L.resize(0, 0);

	// Gradient Term
	// Build S
	matrix S(numOfNode, numOfNode);
	//tbb::concurrent_vector<triplet> SCoef;
	std::vector<triplet> SCoeff;
	// set identity
	S.setIdentity();
	
	// mark each edge as -1.0f
	//tbb::serial::parallel_for(0, (int)mEdges.size(), [&](const int i)
	for (int i = 0; i < (int)mEdges.size(); ++i)
	{
		auto& edge = getEdge(i);

		auto f1 = edge.nodeIds[0];
		auto f2 = edge.nodeIds[1];

		SCoeff.push_back(triplet(f1, f2, -1.0));
		SCoeff.push_back(triplet(f2, f1, -1.0));
	}//);
	S.setFromTriplets(SCoeff.begin(), SCoeff.end());
	SCoeff.clear();

	A += 2 * (S * S);
	
	// Clear S
	S.resize(0, 0);

	// Build B
	// For each face that is labeled assign Xc(strokeId)
	matrix B(numOfNode, numOfStrokes);
	//tbb::concurrent_vector<triplet> BCoef;
	std::vector<triplet> BCoeff;
	//tbb::serial::parallel_for(0, (int)inStrokes.size(), [&](const int strokeId)
	for (int strokeId = 0; strokeId < (int)inStrokes.size(); ++strokeId)
	{
		auto& stroke = inStrokes[strokeId];
		//tbb::serial::parallel_for(0, (int)stroke.size(), [&](const int nodeId)
		for (auto nodeId : stroke)
		{
			for (int k = 0; k < numOfStrokes; ++k)
			{
				BCoeff.push_back(triplet(nodeId, k, Xc.coeff(strokeId, k)));
			}
		}//);
	}//);
	B.setFromTriplets(BCoeff.begin(), BCoeff.end());
	BCoeff.clear();

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

	Eigen::SparseMatrix<double> sparse = A;

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
			double value = std::abs(eval.coeff(i, k));

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
