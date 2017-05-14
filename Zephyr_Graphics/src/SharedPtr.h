#ifndef SHARED_POINTER_H
#define SHARED_POINTER_H

// Specialized std::shared_ptr that call directx12 Release() instead of delete

#include "stdfx.h"

// this will only call release if an object exists (prevents exceptions calling release on non existant objects)
#define SAFE_RELEASE(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }

namespace Zephyr
{
	template <typename T>
	struct Releaser {
		void operator()(T* p) {
			if (nullptr != p)
			{
				SAFE_RELEASE(p)
			}
		};
	};

	template <typename T>
	struct SharedPtr
	{
		SharedPtr(T* p = nullptr)
		{
			mPtr = std::shared_ptr<T>(p, Releaser<T>());
		}
		~SharedPtr()
		{
			mPtr.reset();
		}

		void reset(T* p)
		{
			auto ptr = std::shared_ptr<T>(p, Releaser<T>());
			mPtr = ptr;
		}

		void reset()
		{
			mPtr.reset();
		}

		T* get() const
		{
			return mPtr.get();
		}

		T* operator->()
		{
			return mPtr.operator->();
		}

		bool operator==(T* p)
		{
			return mPtr == p;
		}

		// HACK: couldn't figure out operator==
		bool isNull()
		{
			return nullptr == mPtr;
		}

		std::shared_ptr<T> mPtr;
	};
}

#endif
