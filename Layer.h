#pragma once

#include "Tensor.h"

#include <math.h>
#include <algorithm>
#include <assert.h>

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

inline float Lerp( float t, float a, float b )
{
	return a + t * (b - a);
}

inline float Random( float _min, float _max )
{
	float u = rand() / (float)RAND_MAX;
	return Lerp( u, _min, _max );
}

class Layer
{
public:
	Layer( uint32_t _numInputs, uint32_t _numOutputs ) : m_NumInputs( _numInputs ),	m_NumOutputs( _numOutputs ) {}
	virtual ~Layer() {}
	virtual void Forward( const Tensor& _in, Tensor& _out ) = 0;
	virtual void ClearWeightDeltas() = 0;
	virtual void ApplyWeightDeltas( float _learningRate ) = 0;
	virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients/*in*/, Tensor& _inputGradients /*out*/) = 0;
	virtual const Tensor& GetOutput() const = 0;

	inline uint32_t GetNumInputs() const { return m_NumInputs; }
	inline uint32_t GetNumOutputs() const { return m_NumOutputs; }
protected:
	uint32_t m_NumInputs, m_NumOutputs;

};

template <typename Activation>
class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer( uint32_t _numInputs, uint32_t _numOutputs /*number of neurons*/) :
		Layer( _numInputs, _numOutputs )
	{
		m_Weights.resize( _numInputs * _numOutputs );
		m_DeltaWeights.resize( _numInputs * _numOutputs );

		m_Biases.resize( _numOutputs );
		m_DeltaBiases.resize( _numOutputs );

		m_NetInputs.resize( _numOutputs );
		m_Activations.resize( _numOutputs );

		// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=each%20in%20turn.-,Xavier%20Weight%20Initialization,of%20inputs%20to%20the%20node.&text=We%20can%20implement%20this%20directly%20in%20Python.
		float xavierWeightRange = 1.0f / sqrtf( (float)_numInputs );

		std::generate( m_Weights.begin(), m_Weights.end(), [&]() { return Random( -xavierWeightRange, xavierWeightRange ); } );
		std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
	}

	virtual void Forward( const Tensor& _in, Tensor& _out ) override
	{
		_out.resize( m_NumOutputs );

		#pragma omp parallel for
		for( int i = 0 ; i < (int)m_NumOutputs ; ++i )
		{
			float netSum = m_Biases[i];

			uint32_t weightIdx = i * m_NumInputs;

			for( uint32_t j = 0 ; j < m_NumInputs ; ++j )
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
		for( int i = 0 ; i < (int)m_NumOutputs ; ++i )
		{
			m_Biases[i] -= _learningRate * m_DeltaBiases[i];

			for( uint32_t j = 0 ; j < m_NumInputs ; ++j )
			{
				m_Weights[j + i * m_NumInputs] -= _learningRate * m_DeltaWeights[j + i * m_NumInputs];
			}
		}
	}

	virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
	{
		_inputGradients.resize( m_NumInputs );
		std::fill( _inputGradients.begin(), _inputGradients.end(), 0.0f );

		#pragma omp parallel for
		for( int i = 0 ; i < (int)m_NumOutputs ; ++i )
		{
			float dA_dN = Activation::ComputeDerivative( m_NetInputs[i], m_Activations[i] );
			float dE_dN = _outputGradients[i] * dA_dN;

			m_DeltaBiases[i] += dE_dN; /*dN_dB is ignored because it is 1*/


			for( uint32_t j = 0 ; j < m_NumInputs ; ++j )
			{
				float dN_dW = _layerInputs[j];
				m_DeltaWeights[j + i * m_NumInputs] += dE_dN * dN_dW;
				
				_inputGradients[j] += dE_dN * m_Weights[j + i * m_NumInputs];
			}
			
		}

	}

	virtual const Tensor& GetOutput() const override { return m_Activations; }

private:
	std::vector<float> m_Weights;
	std::vector<float> m_Biases;
	std::vector<float> m_DeltaWeights;
	std::vector<float> m_DeltaBiases;
	Tensor m_NetInputs;
	Tensor m_Activations;
};
