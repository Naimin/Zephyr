#ifndef ZEPHYR_UI_H
#define ZEPHYR_UI_H

#include <string>
#include <Eigen/Eigen>
#include <unordered_map>

#include <nana/gui/wvl.hpp> 
#include <nana/gui/widgets/button.hpp>

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

		std::shared_ptr<nana::button> getButton(const std::string& buttonName);
		std::shared_ptr<nana::button> createButton(std::shared_ptr<nana::nested_form> parent, const std::string& buttonName, nana::rectangle& rect);

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
		std::unordered_map<std::string, std::shared_ptr<nana::button>> mButtonList;

		// mouse event data
		float mZoom;
		Eigen::Vector2i mMousePosition;
	};
}

#endif