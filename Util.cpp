#include "Util.h"

#include <stdio.h>
#include <stdarg.h>
//#include <varargs.h>

#include <windows.h>

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
