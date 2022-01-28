#pragma once

// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

#include "Layer.h"
#include <cmath>

namespace ToyDNN
{
	class BaseActivationLayer : public Layer
	{
	public:
		virtual void Setup( const TensorShape& _previousLayerOutputShape ) override
		{
			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = _previousLayerOutputShape;

			m_Activations.resize( m_OutputShape.Size() );
		}

		virtual void ClearWeightDeltas() override {}
		virtual void ApplyWeightDeltas( float _learningRate ) override {}

		virtual const Tensor& GetOutput() const override { return m_Activations; }

	protected:
		mutable Tensor m_Activations;
	};

	//====================================================

	class Relu : public BaseActivationLayer
	{
	public:

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				_out[i] = std::abs( _in[i] );

				//TODO if( isTraining )
				{
					m_Activations[i] = _out[i];
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t n = m_OutputShape.Size();
			_inputGradients.resize( n, 0.0f );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				float gradient = _layerInputs[i] > 0.0f ? 1.0f : 0.0f;

				_inputGradients[i] = _outputGradients[i] * gradient;
			}

		}
	};

	//====================================================

	class LeakyRelu : public BaseActivationLayer
	{
	public:
		LeakyRelu( float _leak=0.01f) : m_Leak( _leak )
		{
			assert( _leak > 0.0f );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				_out[i] = _in[i] > 0.0f ? _in[i] : _in[i] * m_Leak;

				//TODO if( isTraining )
				{
					m_Activations[i] = _out[i];
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t n = m_OutputShape.Size();
			_inputGradients.resize( n );

		#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				float gradient = _layerInputs[i] > 0.0f ? 1.0f : m_Leak;

				_inputGradients[i] = _outputGradients[i] * gradient;
			}

		}

	private:
		float m_Leak;
	};

	//====================================================

	class Sigmoid : public BaseActivationLayer
	{
	public:

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				_out[i] = 1.0f / (1.0f + std::exp( -_in[i] ));

				//TODO if( isTraining )
				{
					m_Activations[i] = _out[i];
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t n = m_OutputShape.Size();
			_inputGradients.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				//This one is a bit special
				//dsigmoid(x)/dx = sigmoid(x) * (1 - sigmoid(x))

				float gradient = m_Activations[i] * (1.0f - m_Activations[i]);

				_inputGradients[i] = _outputGradients[i] * gradient;
			}

		}
	};

	//====================================================

	class Tanh : public BaseActivationLayer
	{
	public:

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				_out[i] = std::tanh( _in[i] );

				//TODO if( isTraining )
				{
					m_Activations[i] = _out[i];
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t n = m_OutputShape.Size();
			_inputGradients.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				//This one is a bit special
				//dtanh(x)/dx = 1 - tanh(x)

				float gradient = 1.0f - m_Activations[i];

				_inputGradients[i] = _outputGradients[i] * gradient;
			}

		}
	};

	class SoftMax : public BaseActivationLayer
	{
	public:
		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			float alpha = *std::max_element( _in.begin(), _in.end() );
			float denominator = 0;

			for( int i = 0; i < n; i++ )
			{
				_out[i] = std::exp( _in[i] - alpha );
				denominator += _out[i];
			}

			float numerator = 1.0f / denominator;

			for( int i = 0; i < n; i++ )
			{
				_out[i] *= numerator;
				m_Activations[i] = _out[i];
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t n = m_OutputShape.Size();
			_inputGradients.resize( n );

			float* df = (float*)alloca( sizeof( float ) * n );
			memset( df, 0, sizeof( float ) * n );

			for( int j=0; j < n ; ++j )
			{
				for( int k=0; k < n ; ++k )
				{
					float f = (k == j) ? 1.0f : 0.0f;
					df[k] = m_Activations[j] * (f - m_Activations[j]);
				}

				for( int k = 0; k < n ; ++k )
				{
					_inputGradients[j] += _outputGradients[k] * df[k];
				}
			}
		}
	};
}
