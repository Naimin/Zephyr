#include "AppEvents.h"
#include <nana/gui/filebox.hpp>
#include <BasicRenderPass.h>

// external algorithms
#include <Segmentation/TriDualGraph.h>
#include <IO/MeshConverter.h>
#include <Decimate/Decimate.h>

Zephyr::AppEvents::AppEvents(App* pApp, UI * pUI) : mpApp(pApp), mpUI(pUI)
{
}

Zephyr::AppEvents::~AppEvents()
{
}

bool Zephyr::AppEvents::initialize()
{
	// Buttons creation and events
	auto loadBtn = mpUI->createButton(mpUI->getNestedForm(), "Load", nana::rectangle{ 0, 0, 120, 20 });
	setupLoadButtonEvents(loadBtn);

	auto segmentBtn = mpUI->createButton(mpUI->getNestedForm(), "Segment", nana::rectangle{ 0, 30, 120, 20 });
	setupSegmentButtonEvents(segmentBtn);

	auto greedyDecimateBtn = mpUI->createButton(mpUI->getNestedForm(), "Greedy Decimation", nana::rectangle{ 0, 60, 120, 20 });
	setupGreedyDecimationButtonEvents(greedyDecimateBtn);

	auto randomDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "Random Decimation", nana::rectangle{ 0, 90, 120, 20 });
	setupRandomDecimationButtonEvents(randomDecimationBtn);

	return true;
}

void Zephyr::AppEvents::setupLoadButtonEvents(std::shared_ptr<nana::button> pButton)
{
	pButton->events().click([&] {
		nana::filebox fb(*mpUI->getNestedForm(), true);
		fb.add_filter("Model File", "*.obj;*.ply;*.fbx");
		fb.add_filter("All Files", "*.*");

		if (fb())
		{
			auto modelPath = fb.file();
			std::cout << modelPath << std::endl;

			//Common::OpenMeshMesh omesh(modelPath);
			//omesh.exports("D:\\sandbox\\mesh_openmesh.obj");

			auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
			pRenderPass->loadModel(modelPath);

			//pRenderPass->loadModel(modelPath);
			//Algorithm::OpenMeshMesh mesh(pRenderPass->getModel()->getMesh(0));
		}
	});
}

void Zephyr::AppEvents::setupSegmentButtonEvents(std::shared_ptr<nana::button> pButton)
{
	pButton->events().click([&] {
		auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		auto mesh = pModel->getMesh(0);
		Algorithm::TriDualGraph graph(&mesh);

		std::vector<std::vector<int>> input;
		input.push_back(std::vector<int>());
		input.back().push_back(10000);

		input.push_back(std::vector<int>());
		input.back().push_back(1);

		input.push_back(std::vector<int>());
		input.back().push_back(20750);

		input.push_back(std::vector<int>());
		input.back().push_back(5000);

		graph.segment(input);
	});
}

void Zephyr::AppEvents::setupGreedyDecimationButtonEvents(std::shared_ptr<nana::button> pButton)
{
	pButton->events().click([&] {
		auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		auto omesh = Common::MeshConverter::ModelToOpenMesh(*pModel);
		auto numOfFaces = omesh.getMesh().n_faces();

		Algorithm::Decimater::decimate(omesh, (unsigned int)(numOfFaces / 2), Algorithm::GREEDY_DECIMATE);

		omesh.exports("D:\\sandbox\\decimatedGreedyMesh.obj");
	});
}

void Zephyr::AppEvents::setupRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton)
{
	pButton->events().click([&] {
		auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		auto omesh = Common::MeshConverter::ModelToOpenMesh(*pModel);
		auto numOfFaces = omesh.getMesh().n_faces();

		Algorithm::Decimater::decimate(omesh, (unsigned int)(numOfFaces / 2), Algorithm::RANDOM_DECIMATE);

		omesh.exports("D:\\sandbox\\decimatedRandomMesh.obj");
	});
}
