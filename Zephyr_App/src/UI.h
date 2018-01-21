#ifndef ZEPHYR_UI_H
#define ZEPHYR_UI_H

#include <string>
#include <Eigen/Eigen>
#include <unordered_map>

#include <nana/gui/wvl.hpp> 
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/slider.hpp>

namespace Zephyr
{
	class App;


	class UI 
	{
	public:
		UI(const std::string& windowTitle, App* pApp);
		virtual ~UI();

		bool initialize();
		void show();
		std::shared_ptr<nana::form> getForm();
		std::shared_ptr<nana::nested_form> getNestedForm();

		std::shared_ptr<nana::label> getLabel(const std::string& labelName);
		std::shared_ptr<nana::label> createLabel(std::shared_ptr<nana::nested_form> parent, const std::string& labelName, nana::rectangle& rect);
		void updateCaption(const std::string& widgetName, const std::string caption);

		std::shared_ptr<nana::button> getButton(const std::string& buttonName);
		std::shared_ptr<nana::button> createButton(std::shared_ptr<nana::nested_form> parent, const std::string& buttonName, nana::rectangle& rect);

		std::shared_ptr<nana::slider> getSlider(const std::string& sliderName);
		std::shared_ptr<nana::slider> createSlider(std::shared_ptr<nana::nested_form> parent, const std::string& sliderName, nana::rectangle& rect);

	protected:
		// mouse events
		void setupMouseMouseEvent();
		void setupMouseClickEvent();
		void setupMouseWheelEvent();

	private:
		App* mpApp;

		std::shared_ptr<nana::form> mpForm;
		std::shared_ptr<nana::nested_form> mpNfm;

		// buttons
		std::unordered_map <std::string, std::shared_ptr<nana::widget>> mWidgetList;

		// mouse event data
		float mZoom;
		Eigen::Vector2i mMousePosition;
	};
}

#endif