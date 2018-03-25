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
	const unsigned int width = 140;
	int yPos = 5;
	int margin = 5;
	// Buttons creation and events
	auto loadBtn = mpUI->createButton(mpUI->getNestedForm(), "Load", nana::rectangle{ margin, yPos, width - margin, 20 });
	setupLoadButtonEvents(loadBtn);

	yPos += 30;
	auto segmentBtn = mpUI->createButton(mpUI->getNestedForm(), "Segment", nana::rectangle{ margin, yPos, width - margin, 20 });
	setupSegmentButtonEvents(segmentBtn);

	// decimation Slider uses this label
	auto decimationLabel = mpUI->createLabel(mpUI->getNestedForm(), "Decimate by: 50%", nana::rectangle{ margin, 180, width - margin, 20 });
	// the different decimation buttons uses this slider
	auto decimationSlider = mpUI->createSlider(mpUI->getNestedForm(), "Decimation Slider", nana::rectangle{ margin, 195, width - margin, 20 });
	setupDecimationSlider(decimationSlider, decimationLabel);

	// bin size slider uses this labe;
	auto binLabel = mpUI->createLabel(mpUI->getNestedForm(), "Bin Size: 8", nana::rectangle{ margin, 220, width - margin, 20 });
	auto binSlider = mpUI->createSlider(mpUI->getNestedForm(), "Bin Slider", nana::rectangle{ margin, 235, width - margin, 20 });
	setupBinSlider(binSlider, binLabel);

	yPos += 30;
	auto greedyDecimateBtn = mpUI->createButton(mpUI->getNestedForm(), "Greedy Decimation", nana::rectangle{ margin, yPos, width - margin, 20 });
	setupGreedyDecimationButtonEvents(greedyDecimateBtn, decimationSlider, binSlider);

	yPos += 30;
	auto randomDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "Random Decimation", nana::rectangle{ margin, yPos, width - margin, 20 });
	setupRandomDecimationButtonEvents(randomDecimationBtn, decimationSlider, binSlider);

	yPos += 30;
	auto vertexDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "Vertex Random Deci", nana::rectangle{ margin, yPos, width - margin, 20 });
	setupVertexRandomDecimationButtonEvents(vertexDecimationBtn, decimationSlider, binSlider);

	yPos += 30;
	auto gpuDecimationBtn = mpUI->createButton(mpUI->getNestedForm(), "GPU Random Deci", nana::rectangle{ margin, yPos, width - margin, 20 });
	setupGPURandomDecimationButtonEvents(gpuDecimationBtn, decimationSlider, binSlider);

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

void Zephyr::AppEvents::setupGreedyDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::slider> pBinSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, pBinSlider, Algorithm::GREEDY_DECIMATE);
}

void Zephyr::AppEvents::setupRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::slider> pBinSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, pBinSlider, Algorithm::RANDOM_DECIMATE);
}

void Zephyr::AppEvents::setupVertexRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::slider> pBinSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, pBinSlider, Algorithm::GPU_SUPER_VERTEX);
}

void Zephyr::AppEvents::setupGPURandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::slider> pBinSlider)
{
	setupDecimationButtonEvents(pButton, pSlider, pBinSlider, Algorithm::GPU_RANDOM_DECIMATE);
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

void Zephyr::AppEvents::setupBinSlider(std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::label> pLabel)
{
	pSlider->maximum(256);
	pSlider->value(8);
	pLabel->caption("Bin Size: " + std::to_string(pSlider->value()));

	pSlider->scheme().vernier_text_margin = 0; //Modify the margin of the tip label
	pSlider->vernier([](unsigned maximum, unsigned cursor_value)
	{
		return std::to_string(cursor_value); //It should return a UTF-8 string.
	});

	pSlider->events().value_changed([&, pLabel](const nana::arg_slider& slider)
	{
		if (slider.widget.value() < 1)
		{
			slider.widget.value(1);
		}
		pLabel->caption("Bin Size: " + std::to_string(slider.widget.value()));
	});
}


void Zephyr::AppEvents::setupDecimationButtonEvents(std::shared_ptr<nana::button> pButton, std::shared_ptr<nana::slider> pSlider, std::shared_ptr<nana::slider> pBinSlider, Algorithm::DecimationType decimationType)
{
	pButton->events().click([&, pSlider, pBinSlider, decimationType] {
		auto pRenderPass = dynamic_cast<Graphics::BasicRenderPass*>(mpApp->getGraphicsEngine()->getRenderer()->getRenderPass(DEFAULT_RENDERPASS_NAME).get());
		auto pModel = pRenderPass->getModel();

		if (nullptr == pModel)
			return;

		// get the export path
		auto exportPath = getExportPath();

		if (exportPath.empty())
			return;

		// get the target face count number
		float percentage = 1.0f - (pSlider->value() / (float)pSlider->maximum());
		int binSize = pBinSlider->value();

		auto omesh = Common::MeshConverter::ModelToOpenMesh(*pModel);
		auto numOfFaces = omesh.getMesh().n_faces();
		unsigned int targetFaceCount = (unsigned int)(numOfFaces * percentage);

		// Decimate
		if (Algorithm::DecimationType::GPU_RANDOM_DECIMATE == decimationType || Algorithm::DecimationType::GPU_SUPER_VERTEX == decimationType)
		{
			GPU::decimate(omesh, targetFaceCount, binSize, decimationType);
		}
		else
		{
			Algorithm::Decimater::decimate(omesh, targetFaceCount, binSize, decimationType);
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
