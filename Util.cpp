#include "pch.h"
#include "Util.h"

#include <stdio.h>
#include <stdarg.h>

#include <windows.h>
#undef min
#undef max

namespace ToyDNN
{
	Random g_Random;

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

	void ComputeMeanAndVariance( const std::vector< Scalar >& _data, Scalar& _mean, Scalar& _variance )
	{
		_mean = Scalar(0);
		_variance = Scalar( 0 );

		for( Scalar s : _data )
		{
			_mean += s;
		}

		_mean /= (Scalar)_data.size();

		for( Scalar s : _data )
		{
			Scalar d = s - _mean;
			_variance += d * d;
		}

		_variance /= (Scalar)_data.size();
	}

	Scalar PercentageOfZeroValues( const std::vector< Scalar >& _data )
	{
		uint32_t numZeroes = 0;

		for( Scalar s : _data )
		{
			if( s == Scalar( 0 ) )
				++numZeroes;
		}

		return Scalar( 100 * numZeroes ) / Scalar( _data.size() );
	}
}