#pragma once

#include <math.h>

// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

namespace Activation
{

	class Identity
	{
	public:
		static float Compute( float x )
		{
			return x;
		}

		static float ComputeDerivative( float x, float y )
		{
			return 1.0f;
		}
	};


	class Sigmoid
	{
	public:
		static float Compute( float x )
		{
			return 1.0f / (1.0f + expf( -x ));
		}

		static float ComputeDerivative( float x, float y )
		{
			return y * (1.0f - y);
		}
	};


	class Tanh
	{
	public:
		static float Compute( float x )
		{
			return tanhf( x );
		}

		static float ComputeDerivative( float x, float y )
		{
			return 1.0f - y;
		}
	};

	class Relu
	{
	public:
		static float Compute( float x )
		{
			return x < 0.0f ? 0.0f : x;
		}

		static float ComputeDerivative( float x, float y )
		{
			return x < 0.0f ? 0.0f : 1.0f;
		}
	};

	class LeakyRelu
	{
	public:
		static float Compute( float x )
		{
			return x < 0.0f ? 0.01f * x : x;
		}

		static float ComputeDerivative( float x, float y )
		{
			return x < 0.0f ? 0.01f : 1.0f;
		}
	};
}
