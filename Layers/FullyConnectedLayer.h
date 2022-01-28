#pragma once

#include "Layer.h"

template <typename Activation>
class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer( uint32_t _numNeurons )
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

		m_NetInputs.resize( m_OutputShape.m_SX );
		m_Activations.resize( m_OutputShape.m_SX );

		// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=each%20in%20turn.-,Xavier%20Weight%20Initialization,of%20inputs%20to%20the%20node.&text=We%20can%20implement%20this%20directly%20in%20Python.
		float xavierWeightRange = 1.0f / sqrtf( (float)m_InputShape.m_SX );

		std::generate( m_Weights.begin(), m_Weights.end(), [&]() { return Random( -xavierWeightRange, xavierWeightRange ); } );
		std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
	}

	virtual void Forward( const Tensor& _in, Tensor& _out ) const override
	{
		_out.resize( m_OutputShape.m_SX );

	#pragma omp parallel for
		for( int i = 0 ; i < (int)m_OutputShape.m_SX ; ++i )
		{
			float netSum = m_Biases[i];

			uint32_t weightIdx = i * m_InputShape.m_SX;

			for( uint32_t j = 0 ; j < m_InputShape.m_SX ; ++j )
			{
				netSum += _in[j] * m_Weights[weightIdx + j];
			}

			_out[i] = Activation::Compute( netSum );

			//TODO if( isTraining )
			{
				m_NetInputs[i] = netSum;
				m_Activations[i] = _out[i];
			}
		}
	}

	virtual void ClearWeightDeltas() override
	{
		std::fill( m_DeltaWeights.begin(), m_DeltaWeights.end(), 0.0f );
		std::fill( m_DeltaBiases.begin(), m_DeltaBiases.end(), 0.0f );
	}

	virtual void ApplyWeightDeltas( float _learningRate ) override
	{
	#pragma omp parallel for
		for( int i = 0 ; i < (int)m_OutputShape.m_SX ; ++i )
		{
			m_Biases[i] -= _learningRate * m_DeltaBiases[i];

			for( uint32_t j = 0 ; j < m_InputShape.m_SX ; ++j )
			{
				m_Weights[j + i * m_InputShape.m_SX] -= _learningRate * m_DeltaWeights[j + i * m_InputShape.m_SX];
			}
		}
	}

	virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
	{
		_inputGradients.resize( m_InputShape.m_SX, 0.0f );

	#pragma omp parallel for
		for( int i = 0 ; i < (int)m_OutputShape.m_SX ; ++i )
		{
			float dA_dN = Activation::ComputeDerivative( m_NetInputs[i], m_Activations[i] );
			float dE_dN = _outputGradients[i] * dA_dN;

			m_DeltaBiases[i] += dE_dN; /*dN_dB is ignored because it is 1*/

			for( uint32_t j = 0 ; j < m_InputShape.m_SX ; ++j )
			{
				float dN_dW = _layerInputs[j];
				m_DeltaWeights[j + i * m_InputShape.m_SX] += dE_dN * dN_dW;

				_inputGradients[j] += dE_dN * m_Weights[j + i * m_InputShape.m_SX];
			}

		}

	}

	virtual const Tensor& GetOutput() const override { return m_Activations; }

private:
	std::vector<float> m_Weights;
	std::vector<float> m_Biases;
	std::vector<float> m_DeltaWeights;
	std::vector<float> m_DeltaBiases;
	mutable Tensor m_NetInputs;
	mutable Tensor m_Activations;
};
