#pragma once

#include "Layer.h"

namespace ToyDNN
{

	class FullyConnected : public WeightsAndBiasesLayer
	{
	public:
		FullyConnected( uint32_t _numNeurons=0 )
		{
			m_OutputShape = TensorShape( _numNeurons, 1, 1 );
		}

		FullyConnected( const TensorShape& _outputShape )
		{
			m_OutputShape = _outputShape;
		}

		virtual LayerType GetType() const override { return LayerType::FullyConnected; }
		virtual const char* GetName() const override { return "FullyConnected"; }

		virtual void Setup( const TensorShape& _previousLayerOutputShape, uint32_t _outputPadding ) override
		{
			m_InputShape = _previousLayerOutputShape;

			uint32_t inputSize = m_InputShape.SizeWithoutPadding();
			uint32_t outputSize = m_OutputShape.SizeWithoutPadding();

			m_Weights.resize( inputSize * outputSize );
			m_WeightGradients.resize( m_Weights.size() );

			m_Biases.resize( outputSize );
			m_BiasGradients.resize( outputSize );

			//WeightInit::Xavier( m_InputShape.Size(), m_OutputShape.Size(), m_Weights );
			WeightInit::He( inputSize, outputSize, m_Weights );

			std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			uint32_t inputSize = m_InputShape.Size();
			uint32_t outputSize = m_OutputShape.Size();

			//#pragma omp parallel for
			for( int i = 0 ; i < (int)outputSize ; ++i )
			{
				Scalar netSum = m_Biases[i];

				uint32_t weightIdx = i * inputSize;

				for( uint32_t j = 0 ; j < inputSize ; ++j )
				{
					netSum += _in[j] * m_Weights[weightIdx + j];
				}

				_out[i] = netSum;
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			uint32_t inputSize = m_InputShape.Size();
			uint32_t outputSize = m_OutputShape.Size();

			std::fill( _inputGradients.begin(), _inputGradients.end(), Scalar( 0.0 ) );

			//#pragma omp parallel for
			for( int i = 0 ; i < (int)outputSize ; ++i )
			{
				Scalar dE_dN = _outputGradients[i];

				m_BiasGradients[i] += dE_dN; /*dN_dB is ignored because it is 1*/

				for( uint32_t j = 0 ; j < inputSize ; ++j )
				{
					Scalar dN_dW = _layerInputs[j];
					m_WeightGradients[j + i * inputSize] += dE_dN * dN_dW;

					_inputGradients[j] += dE_dN * m_Weights[j + i * inputSize];
				}

			}

		}

	};
}
