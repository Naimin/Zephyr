#include "AppEvents.h"
#include <nana/gui/filebox.hpp>
#include <nana/gui/widgets/slider.hpp>
#include <BasicRenderPass.h>

#include <boost/filesystem/path.hpp>

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

	auto decimationLabel = mpUI->createLabel(mpUI->getNestedForm(), "Decimate by: 50%", nana::rectangle{ 0, 120, 120, 20 });

	auto decimationSlider = mpUI->createSlider(mpUI->getNestedForm(), "Decimation Slider", nana::rectangle{ 0, 140, 120, 20 });
	setupDecimationSlider(decimationSlider, decimationLabel);

	auto greedyDecimateBtn = mpUI->createButton(mpUI->getNestedForm(), "Greedy Decimation", nana::rectangle{ 0, 60, 120, 20 });
	setupGreedyDecimationButtonEvents(greedyDecimateBtn, decimationSlider);

	auto randomDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "Random Decimation", nana::rectangle{ 0, 90, 120, 20 });
	setupRandomDecimationButtonEvents(randomDecimationBtn, decimationSlider);

	return true;
}

void Zephyr::AppEvents::setupLoadButtonEvents(std::shared_ptr<nana::button> pButton)
{
	pButton->events().click([&] {
		nana::filebox fb(*mpUI->getNestedForm(), true);
		fb.add_filter("Model File (.obj, .ply, .fbx)", "*.obj;*.ply;*.fbx");
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

void Zephyr::AppEvents::setupGreedyDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider)
{
	pButton->events().click([&, pSlider] {
		auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		// get the export path
		auto exportPath = getExportPath();

		// get the target face count number
		float percentage = 1.0f - (pSlider->value() / (float)pSlider->maximum());

		auto omesh = Common::MeshConverter::ModelToOpenMesh(*pModel);
		auto numOfFaces = omesh.getMesh().n_faces();

		// Decimate
		Algorithm::Decimater::decimate(omesh, (unsigned int)(numOfFaces * percentage), Algorithm::GREEDY_DECIMATE);

		omesh.exports(exportPath);
	});
}

void Zephyr::AppEvents::setupRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider)
{
	pButton->events().click([&, pSlider] {
		auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		// get the export path
		auto exportPath = getExportPath();

		// get the target face count number
		float percentage = 1.0f - (pSlider->value() / (float)pSlider->maximum());

		auto omesh = Common::MeshConverter::ModelToOpenMesh(*pModel);
		auto numOfFaces = omesh.getMesh().n_faces();

		// Decimate
		Algorithm::Decimater::decimate(omesh, (unsigned int)(numOfFaces * percentage), Algorithm::RANDOM_DECIMATE);

		omesh.exports(exportPath);
	});
}

void Zephyr::AppEvents::setupDecimationSlider(std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::label> pLabel)
{
	pSlider->maximum(100);
	pSlider->value(50);

	pSlider->scheme().vernier_text_margin = 0; //Modify the margin of the tip label
	pSlider->vernier([](unsigned maximum, unsigned cursor_value)
	{
		return std::to_string(cursor_value * 100 / maximum) + "%"; //It should return a UTF-8 string.
	});

	pSlider->events().value_changed([&, pLabel](const nana::arg_slider& slider)
	{
		pLabel->caption("Decimate by: " + std::to_string(slider.widget.value() * 100 / slider.widget.maximum()) + "%");
	});

}

std::string Zephyr::AppEvents::getExportPath()
{
	nana::filebox fb(*mpUI->getNestedForm(), false);
	fb.add_filter("Object File (.obj)", "*.obj");
	fb.add_filter("PLY File (.ply)", "*.ply");
	fb.add_filter("FilmBox File (.fbx)", "*.fbx");
	fb.add_filter("Model File (.obj, .ply, .fbx)", "*.obj;*.ply;*.fbx");
	fb.add_filter("All Files", "*.*");

	std::string exportPath;
	if (fb())
		exportPath = fb.file();

	return exportPath;
}
