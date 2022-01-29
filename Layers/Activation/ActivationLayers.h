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
		virtual void ApplyWeightDeltas( Scalar _learningRate ) override {}

		virtual const Tensor& GetOutput() const override { return m_Activations; }

	protected:
		mutable Tensor m_Activations;
	};

	//====================================================

	class Relu : public BaseActivationLayer
	{
	public:
		virtual LayerType GetType() const override { return LayerType::Relu; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				_out[i] = std::max( _in[i], Scalar(0.0) );

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
				Scalar gradient = _layerInputs[i] > Scalar(0.0) ? Scalar(1.0) : Scalar(0.0);

				_inputGradients[i] = _outputGradients[i] * gradient;
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
				Scalar gradient = _layerInputs[i] > Scalar(0.0) ? Scalar(1.0) : m_Leak;

				_inputGradients[i] = _outputGradients[i] * gradient;
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

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)n ; ++i )
			{
				_out[i] = Scalar(1.0) / (Scalar(1.0) + std::exp( -_in[i] ));

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

				Scalar gradient = m_Activations[i] * (Scalar(1.0) - m_Activations[i]);

				_inputGradients[i] = _outputGradients[i] * gradient;
			}

		}
	};

	//====================================================

	class Tanh : public BaseActivationLayer
	{
	public:
		virtual LayerType GetType() const override { return LayerType::Tanh; }

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

				Scalar gradient = Scalar(1.0) - m_Activations[i] * m_Activations[i];

				_inputGradients[i] = _outputGradients[i] * gradient;
			}

		}
	};

	class SoftMax : public BaseActivationLayer
	{
	public:
		virtual LayerType GetType() const override { return LayerType::SoftMax; }

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t n = m_OutputShape.Size();
			_out.resize( n );

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
				m_Activations[i] = _out[i];
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t n = m_OutputShape.Size();
			_inputGradients.resize( n );

			Scalar* df = (Scalar*)alloca( sizeof( Scalar ) * n );
			memset( df, 0, sizeof( Scalar ) * n );

			for( uint32_t j=0; j < n ; ++j )
			{
				for( uint32_t k=0; k < n ; ++k )
				{
					Scalar f = (k == j) ? Scalar(1.0) : Scalar(0.0);
					df[k] = m_Activations[j] * (f - m_Activations[j]);
				}

				for( uint32_t k = 0; k < n ; ++k )
				{
					_inputGradients[j] += _outputGradients[k] * df[k];
				}
			}
		}
	};
}
