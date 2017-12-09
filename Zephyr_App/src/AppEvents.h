#ifndef ZEPHYR_APP_EVENTS_H
#define ZEPHYR_APP_EVENTS_H

#include "App.h"
#include "UI.h"

namespace Zephyr
{
	class AppEvents
	{
		public:
			AppEvents(App* pApp, UI* pUI);
			virtual ~AppEvents();

			bool initialize();

		protected:
			virtual void setupLoadButtonEvents(std::shared_ptr<nana::button> pButton);
			virtual void setupSegmentButtonEvents(std::shared_ptr<nana::button> pButton);
			virtual void setupGreedyDecimationButtonEvents(std::shared_ptr<nana::button> pButton);
			virtual void setupRandomDecimationButtonEvents(std::shared_ptr<nana::button> pButton);
		
		private:
			App* mpApp;
			UI* mpUI;
	};
}

#endif