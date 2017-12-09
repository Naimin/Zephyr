#include "UI.h"
#include "App.h"

#include <nana/gui/widgets/label.hpp>
#include <nana/gui/timer.hpp>

using namespace Zephyr;

Zephyr::UI::UI(const std::string & windowTitle, App * pApp) : mpApp(pApp), mZoom(0)
{
	mpForm.reset( new nana::form(nana::rectangle{ 0, 0, mpApp->getWidth(), mpApp->getHeight() }) );
	mpForm->caption(windowTitle);

	nana::rectangle UIRect = nana::rectangle{ 10, 10, 120, 110 };

	mpNfm.reset(new nana::nested_form(*mpForm, UIRect, nana::form::appear::bald<>()));

	// set the HWND from the UI framework to the App
	mpApp->getHwnd() = reinterpret_cast<HWND>(mpForm->native_handle());

	initialize();
}

Zephyr::UI::~UI()
{
}

bool Zephyr::UI::initialize()
{
	// Mouse events
	setupMouseMouseEvent();
	setupMouseClickEvent();
	setupMouseWheelEvent();

	return true;
}

void Zephyr::UI::show()
{
	mpNfm->show();
	mpForm->show();
}

std::shared_ptr<nana::form> Zephyr::UI::getForm()
{
	return mpForm;
}

std::shared_ptr<nana::nested_form> Zephyr::UI::getNestedForm()
{
	return mpNfm;
}

std::shared_ptr<nana::button> Zephyr::UI::getButton(const std::string & buttonName)
{
	return mButtonList[buttonName];
}

std::shared_ptr<nana::button> Zephyr::UI::createButton(std::shared_ptr<nana::nested_form> parent, const std::string & buttonName, nana::rectangle& rect)
{
	std::shared_ptr<nana::button> pButton(new nana::button(*parent, rect));
	pButton->caption(buttonName);

	mButtonList.insert(std::make_pair(buttonName, pButton));

	return pButton;
}

void Zephyr::UI::setupMouseMouseEvent()
{
	mpForm->events().mouse_move([&](const nana::arg_mouse& mouseEvent)
	{
		int deltaX = mouseEvent.pos.x - mMousePosition[0];
		int deltaY = mouseEvent.pos.y - mMousePosition[1];

		// hold Mid mouse for zoom
		if (mouseEvent.mid_button)
		{
			mpApp->getCamera().zoom((float)-deltaY);
		}

		if (mouseEvent.right_button)
		{
			mpApp->getCamera().rotation((float)deltaX * 0.75f, (float)deltaY * 0.75f);
		}

		mMousePosition[0] = mouseEvent.pos.x;
		mMousePosition[1] = mouseEvent.pos.y;
	});
}

void Zephyr::UI::setupMouseClickEvent()
{
	mpForm->events().click([&](const nana::arg_click& clickEvent)
	{
		if (clickEvent.mouse_args->is_left_button())
		{
			auto pickingRay = mpApp->getCamera().getPickingRay(clickEvent.mouse_args->pos.x, clickEvent.mouse_args->pos.y);
		}
	});
}

void Zephyr::UI::setupMouseWheelEvent()
{
	// one notch of mouse wheel delta is 120
	const float MOUSE_WHEEL_DELTA = 120.0f;
	mpForm->events().mouse_wheel([&](const nana::arg_wheel& mouseWheelEvent)
	{
		auto distance = (mouseWheelEvent.distance / MOUSE_WHEEL_DELTA) * 10;
		distance = mouseWheelEvent.upwards ? distance : -distance;

		mpApp->getCamera().zoom(distance);
	});
}
