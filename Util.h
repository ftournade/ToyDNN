#pragma once

#include <stdlib.h>
#include <istream>
#include <ostream>

#include "Tensor.h"

namespace ToyDNN
{
	void Log( const char* _format, ... );

	inline Scalar Lerp( Scalar t, Scalar a, Scalar b )
	{
		return a + t * (b - a);
	}

	inline Scalar Random( Scalar _min, Scalar _max )
	{
		Scalar u = rand() / (Scalar)RAND_MAX;
		return Lerp( u, _min, _max );
	}

	bool WriteBMP( const char* _filename, bool _grayscale, const Tensor& _pixels, int _width, int _height );

	template< typename T >
	void Write( std::ostream& _stream, const T& _val )
	{
		_stream.write( (const char*)&_val, sizeof( T ) );
	}

	template< typename T >
	void Read( std::istream& _stream, T& _val )
	{
		_stream.read( (char*)&_val, sizeof( T ) );
	}
	
}
