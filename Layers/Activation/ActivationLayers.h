#pragma once

// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

#include "Layer.h"
#include <cmath>

namespace ToyDNN
{
	class BaseActivationLayer : public Layer
	{
	protected:
		BaseActivationLayer() {}
	public:
		virtual void Setup( const TensorShape& _previousLayerOutputShape, uint32_t _outputPadding ) override
		{
			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = _previousLayerOutputShape;
			m_OutputShape.m_SX += _outputPadding - m_InputShape.m_Padding;
			m_OutputShape.m_SY += _outputPadding - m_InputShape.m_Padding;
			m_OutputShape.m_Padding = _outputPadding;
		}
	};

	//====================================================

	class Relu : public BaseActivationLayer
	{
	public:
		virtual LayerType GetType() const override { return LayerType::Relu; }
		virtual const char* GetName() const override { return "Relu"; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			if( m_OutputShape.m_Padding > 0 )
				memset( &_out[0], 0, _out.size() * sizeof( Scalar ) );

			for( uint32_t z = 0 ; z < m_InputShape.m_SZ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						_out[m_OutputShape.PaddedIndex( x, y, z )] = std::max( _in[m_InputShape.Index( x, y, z )], Scalar( 0.0 ) );
					}
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						uint32_t outIdx = m_OutputShape.PaddedIndex( x, y, z );
						uint32_t inIdx = m_InputShape.Index( x, y, z );

						Scalar gradient = _layerInputs[inIdx] >= Scalar( 0.0 ) ? Scalar( 1.0 ) : Scalar( 0.0 );

						_inputGradients[inIdx] = _outputGradients[outIdx] * gradient;
					}
				}
			}
		}
	};

	//====================================================

	class LeakyRelu : public BaseActivationLayer
	{
	public:
		LeakyRelu( Scalar _leak=0.01f) : m_Leak( _leak )
		{
			assert( _leak > Scalar(0.0) );
		}

		virtual LayerType GetType() const override { return LayerType::LeakyRelu; }
		virtual const char* GetName() const override { return "LeakyRelu"; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			if( m_OutputShape.m_Padding > 0 )
				memset( &_out[0], 0, _out.size() * sizeof( Scalar ) );

			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						Scalar in = _in[m_InputShape.Index( x, y, z )];

						_out[m_OutputShape.PaddedIndex( x, y, z )] = in > 0.0f ? in : in * m_Leak;
					}
				}
			}

		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						uint32_t outIdx = m_OutputShape.PaddedIndex( x, y, z );
						uint32_t inIdx = m_InputShape.Index( x, y, z );

						Scalar gradient = _layerInputs[inIdx] >= Scalar( 0.0 ) ? Scalar( 1.0 ) : m_Leak;

						_inputGradients[inIdx] = _outputGradients[outIdx] * gradient;
					}
				}
			}

		}

		virtual void Load( std::istream& _stream ) override
		{
			Read( _stream, m_Leak );
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			Write( _stream, m_Leak );
		}

	private:
		Scalar m_Leak;
	};

	//====================================================

	class Sigmoid : public BaseActivationLayer
	{
	public:
		virtual LayerType GetType() const override { return LayerType::Sigmoid; }
		virtual const char* GetName() const override { return "Sigmoid"; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			if( m_OutputShape.m_Padding > 0 )
				memset( &_out[0], 0, _out.size() * sizeof( Scalar ) );

			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						Scalar in = _in[m_InputShape.Index( x, y, z )];

						_out[m_OutputShape.PaddedIndex( x, y, z )] = Scalar( 1.0 ) / (Scalar( 1.0 ) + std::exp( -in ));
					}
				}
			}

		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						uint32_t outIdx = m_OutputShape.PaddedIndex( x, y, z );
						uint32_t inIdx = m_InputShape.Index( x, y, z );
						Scalar out = GetOutput()[outIdx];

						Scalar gradient = out * (Scalar( 1.0 ) - out);

						_inputGradients[inIdx] = _outputGradients[outIdx] * gradient;
					}
				}
			}

		}
	};

	//====================================================

	class Tanh : public BaseActivationLayer
	{
	public:
		virtual LayerType GetType() const override { return LayerType::Tanh; }
		virtual const char* GetName() const override { return "Tanh"; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			if( m_OutputShape.m_Padding > 0 )
				memset( &_out[0], 0, _out.size() * sizeof( Scalar ) );

			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						Scalar in = _in[m_InputShape.Index( x, y, z )];

						_out[m_OutputShape.PaddedIndex( x, y, z )] = std::tanh( in );
					}
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
					{
						uint32_t outIdx = m_OutputShape.PaddedIndex( x, y, z );
						uint32_t inIdx = m_InputShape.Index( x, y, z );
						Scalar out = GetOutput()[outIdx];

						//This one is a bit special
						//dtanh(x)/dx = 1 - tanh(x)²
						Scalar gradient = Scalar( 1.0 ) - out * out;

						_inputGradients[inIdx] = _outputGradients[outIdx] * gradient;
					}
				}
			}

		}
	};

	class SoftMax : public BaseActivationLayer
	{
	public:
		virtual void FixMeImFailingGradientCheck() = 0;

		virtual LayerType GetType() const override { return LayerType::SoftMax; }
		virtual const char* GetName() const override { return "SoftMax"; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			assert( false );//TODO support padding

			uint32_t n = m_OutputShape.Size();
			
			Scalar alpha = *std::max_element( _in.begin(), _in.end() );
			Scalar denominator = 0.0;

			for( uint32_t i = 0; i < n; i++ )
			{
				_out[i] = std::exp( _in[i] - alpha );
				denominator += _out[i];
			}

			Scalar numerator = Scalar(1.0) / denominator;

			for( uint32_t i = 0; i < n; i++ )
			{
				_out[i] *= numerator;
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			assert( false );//TODO support padding
			
			uint32_t n = m_OutputShape.Size();

			Scalar* df = (Scalar*)alloca( sizeof( Scalar ) * n );
			memset( df, 0, sizeof( Scalar ) * n );

			for( uint32_t j=0; j < n ; ++j )
			{
				for( uint32_t k=0; k < n ; ++k )
				{
					Scalar f = (k == j) ? Scalar(1.0) : Scalar(0.0);
					df[k] = GetOutput()[j] * (f - GetOutput()[j]);
				}

				for( uint32_t k = 0; k < n ; ++k )
				{
					_inputGradients[j] += _outputGradients[k] * df[k];
				}
			}
		}
	};
}
