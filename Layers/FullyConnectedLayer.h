#pragma once

#include "Layer.h"

namespace ToyDNN
{

	class FullyConnected : public WeightsAndBiasesLayer
	{
	public:
		FullyConnected( uint32_t _numNeurons )
		{
			m_OutputShape = TensorShape( _numNeurons, 1, 1 );
		}

		virtual void Setup( const TensorShape& _previousLayerOutputShape ) override
		{
			m_InputShape = TensorShape( _previousLayerOutputShape.Size(), 1, 1 ); //Flatten the input tensor

			m_Weights.resize( m_InputShape.m_SX * m_OutputShape.m_SX );
			m_DeltaWeights.resize( m_Weights.size() );

			m_Biases.resize( m_OutputShape.m_SX );
			m_DeltaBiases.resize( m_OutputShape.m_SX );

			m_Output.resize( m_OutputShape.m_SX );

			// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=each%20in%20turn.-,Xavier%20Weight%20Initialization,of%20inputs%20to%20the%20node.&text=We%20can%20implement%20this%20directly%20in%20Python.
			Scalar xavierWeightRange = Scalar(1.0) / std::sqrt( (Scalar)m_InputShape.m_SX );

			std::generate( m_Weights.begin(), m_Weights.end(), [&]() { return Random( -xavierWeightRange, xavierWeightRange ); } );
			std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			_out.resize( m_OutputShape.m_SX );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)m_OutputShape.m_SX ; ++i )
			{
				Scalar netSum = m_Biases[i];

				uint32_t weightIdx = i * m_InputShape.m_SX;

				for( uint32_t j = 0 ; j < m_InputShape.m_SX ; ++j )
				{
					netSum += _in[j] * m_Weights[weightIdx + j];
				}

				_out[i] = netSum;

				//TODO if( isTraining )
				{
					m_Output[i] = netSum;
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			_inputGradients.resize( m_InputShape.m_SX, 0.0f );

			#pragma omp parallel for
			for( int i = 0 ; i < (int)m_OutputShape.m_SX ; ++i )
			{
				Scalar dE_dN = _outputGradients[i];

				m_DeltaBiases[i] += dE_dN; /*dN_dB is ignored because it is 1*/

				for( uint32_t j = 0 ; j < m_InputShape.m_SX ; ++j )
				{
					Scalar dN_dW = _layerInputs[j];
					m_DeltaWeights[j + i * m_InputShape.m_SX] += dE_dN * dN_dW;

					_inputGradients[j] += dE_dN * m_Weights[j + i * m_InputShape.m_SX];
				}

			}

		}

		virtual const Tensor& GetOutput() const override { return m_Output; }
		
	private:
		mutable Tensor m_Output;
	};
}
