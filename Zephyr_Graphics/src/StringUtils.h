#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <locale>
#include <codecvt>
#include <string>

namespace Zephyr 
{
	namespace Utils
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	}
}


#endif