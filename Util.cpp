#include "pch.h"
#include "Util.h"

#include <stdio.h>
#include <stdarg.h>

#include <windows.h>
#undef min
#undef max

namespace ToyDNN
{
	void Log( const char* _format, ... )
	{
		const int bufferSize = 1024;
		char buffer[bufferSize];

		va_list args;
		va_start( args, _format );
		vsprintf_s( buffer, bufferSize, _format, args );
		va_end( args );

		OutputDebugStringA( buffer );
	}

	void Color::Saturate()
	{
		R = std::min( R, 1.0f );
		R = std::max( R, 0.0f );
		G = std::min( G, 1.0f );
		G = std::max( G, 0.0f );
		B = std::min( B, 1.0f );
		B = std::max( B, 0.0f );
	}

}