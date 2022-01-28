#pragma once

#include <stdlib.h>
#include "Tensor.h"

namespace ToyDNN
{
	void Log( const char* _format, ... );

	inline float Lerp( float t, float a, float b )
	{
		return a + t * (b - a);
	}

	inline float Random( float _min, float _max )
	{
		float u = rand() / (float)RAND_MAX;
		return Lerp( u, _min, _max );
	}

	bool WriteBMP( const char* _filename, bool _grayscale, const Tensor& _pixels, int _width, int _height );
}
