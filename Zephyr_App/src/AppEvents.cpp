#include "AppEvents.h"
#include <nana/gui/filebox.hpp>
#include <nana/gui/widgets/slider.hpp>
#include <BasicRenderPass.h>

#include <boost/filesystem/path.hpp>

// external algorithms
#include <Segmentation/TriDualGraph.h>
#include <IO/MeshConverter.h>
#include <Decimate/Decimate.h>
#include <Algorithm/Decimate.h>
#include <Timer.h>

Zephyr::AppEvents::AppEvents(App* pApp, UI * pUI) : mpApp(pApp), mpUI(pUI)
{
}

Zephyr::AppEvents::~AppEvents()
{
}

bool Zephyr::AppEvents::initialize()
{
	const unsigned int width = 125;
	int yPos = 0;
	// Buttons creation and events
	auto loadBtn = mpUI->createButton(mpUI->getNestedForm(), "Load", nana::rectangle{ 0, yPos, width, 20 });
	setupLoadButtonEvents(loadBtn);

	yPos += 30;
	auto segmentBtn = mpUI->createButton(mpUI->getNestedForm(), "Segment", nana::rectangle{ 0, yPos, width, 20 });
	setupSegmentButtonEvents(segmentBtn);


	// decimation Slider uses this label
	auto decimationLabel = mpUI->createLabel(mpUI->getNestedForm(), "Decimate by: 50%", nana::rectangle{ 0, 170, width, 20 });

	// the different decimation buttons uses this slider
	auto decimationSlider = mpUI->createSlider(mpUI->getNestedForm(), "Decimation Slider", nana::rectangle{ 0, 190, width, 20 });
	setupDecimationSlider(decimationSlider, decimationLabel);

	yPos += 30;
	auto greedyDecimateBtn = mpUI->createButton(mpUI->getNestedForm(), "Greedy Decimation", nana::rectangle{ 0, yPos, width, 20 });
	setupGreedyDecimationButtonEvents(greedyDecimateBtn, decimationSlider);

	yPos += 30;
	auto randomDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "Random Decimation", nana::rectangle{ 0, yPos, width, 20 });
	setupRandomDecimationButtonEvents(randomDecimationBtn, decimationSlider);

	yPos += 30;
	auto vertexDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "Vertex Random Deci", nana::rectangle{ 0, yPos, width, 20 });
	setupVertexRandomDecimationButtonEvents(vertexDecimationBtn, decimationSlider);

	yPos += 30;
	auto gpuDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "GPU Random Deci", nana::rectangle{ 0, yPos, width, 20 });
	setupGPURandomDecimationButtonEvents(gpuDecimationBtn, decimationSlider);

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

			auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
			pRenderPass->loadModel(modelPath);
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
	setupDecimationButtonEvents(pButton, pSlider, Algorithm::GREEDY_DECIMATE);
}

void Zephyr::AppEvents::setupRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, Algorithm::RANDOM_DECIMATE);
}

void Zephyr::AppEvents::setupVertexRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, Algorithm::RANDOM_DECIMATE_VERTEX);
}

void Zephyr::AppEvents::setupGPURandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, Algorithm::GPU_RANDOM_DECIMATE);
}

void Zephyr::AppEvents::setupDecimationSlider(std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::label> pLabel)
{
	pSlider->maximum(100);
	pSlider->value(85);
	pLabel->caption("Decimate by: " + std::to_string(pSlider->value() * 100 / pSlider->maximum()) + "%");

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

void Zephyr::AppEvents::setupDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider, Algorithm::DecimationType decimationType)
{
	pButton->events().click([&, pSlider, decimationType] {
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
		if (Algorithm::DecimationType::GPU_RANDOM_DECIMATE == decimationType)
		{
			Common::Timer timer;

			int collapseCount = -1;
			auto& omeshDecimated = omesh.getMesh();
			auto previousFaceCount = omeshDecimated.n_faces();
			int targetFaceCount = previousFaceCount - (int)(numOfFaces * percentage);
			int binCount = 8;

			std::cout << "GPU Random Decimation..." << std::endl;
			GPU::decimate(omesh, (unsigned int)(numOfFaces * percentage), binCount);

			auto elapseTime = timer.getElapsedTime();

			omeshDecimated.garbage_collection();

			std::cout << "Decimation done in " << elapseTime << " sec" << std::endl;
			std::cout << "Original Face Count: " << previousFaceCount << std::endl;
			std::cout << "Target Face Count: " << targetFaceCount << std::endl;
			std::cout << "Removed Face Count: " << collapseCount << std::endl;
			std::cout << "Decimated Face Count: " << omeshDecimated.n_faces() << std::endl;
			std::cout << "Percentage decimated: " << ((previousFaceCount - omeshDecimated.n_faces()) / (float)previousFaceCount) * 100.0f << " %" << std::endl;
		}
		else
		{
			Algorithm::Decimater::decimate(omesh, (unsigned int)(numOfFaces * percentage), decimationType);
		}

		std::cout << "Saving decimation output to: " << exportPath << std::endl << std::endl;
		omesh.exports(exportPath);
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
