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
	
	class Color
	{
	public:
		Color() {}
		Color( float r, float g, float b ) : R( r ), G( g ), B( b ) {}
		void Saturate();

		inline operator COLORREF() const { return RGB( (int)(R * 255.0f), (int)(G * 255.0f), (int)(B * 255.0f) ); }

		float R, G, B;
	};

	void ComputeMeanAndVariance( const std::vector< Scalar >& _data, Scalar& _mean, Scalar& _variance );
	Scalar PercentageOfZeroValues( const std::vector< Scalar >& _data );
}
