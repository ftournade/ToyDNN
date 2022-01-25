#pragma once

#include "Tensor.h"
#include "ActivationFuncs.h"

#include <algorithm>
#include <assert.h>


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
	Layer() : m_NumInputs( 0 ),	m_NumOutputs( 0 ) {}
	virtual ~Layer() {}
	virtual void Setup( uint32_t _previousLayerNumOutputs ) = 0;
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
	FullyConnectedLayer( uint32_t _numNeurons )
	{
		m_NumOutputs = _numNeurons;
	}

	virtual void Setup( uint32_t _previousLayerNumOutputs ) override 
	{

		m_NumInputs = _previousLayerNumOutputs;

		m_Weights.resize( m_NumInputs * m_NumOutputs );
		m_DeltaWeights.resize( m_NumInputs * m_NumOutputs );

		m_Biases.resize( m_NumOutputs );
		m_DeltaBiases.resize( m_NumOutputs );

		m_NetInputs.resize( m_NumOutputs );
		m_Activations.resize( m_NumOutputs );

		// https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=each%20in%20turn.-,Xavier%20Weight%20Initialization,of%20inputs%20to%20the%20node.&text=We%20can%20implement%20this%20directly%20in%20Python.
		float xavierWeightRange = 1.0f / sqrtf( (float)m_NumInputs );

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
		_inputGradients.resize( m_NumInputs, 0.0f );

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

template <typename Activation>
class Convolution2DLayer : public Layer
{
public:
	Convolution2DLayer( uint32_t _inputSizeX, uint32_t _inputSizeY, uint32_t _numFeatureMaps, uint32_t _kernelSize, uint32_t _stride=1 ) 
	: m_InputSizeX(_inputSizeX), m_InputSizeY(_inputSizeY), m_NumFeatureMaps(_numFeatureMaps), m_KernelSize(_kernelSize), m_Stride(_stride) 
	{}

	virtual void Setup( uint32_t _previousLayerNumOutputs ) override
	{
		assert( _previousLayerNumOutputs % m_InputSizeX == 0 );
		assert( m_InputSizeX * m_InputSizeY == _previousLayerNumOutputs );

		m_NumInputs = _previousLayerNumOutputs;
		m_NumOutputs = (m_InputSizeX - m_KernelSize + 1) * (m_InputSizeY - m_KernelSize + 1) / (m_Stride * m_Stride);
		m_NumOutputs *= m_NumFeatureMaps;

		m_Weights.resize( m_NumFeatureMaps * m_KernelSize * m_KernelSize );
		m_DeltaWeights.resize( m_NumFeatureMaps * m_KernelSize * m_KernelSize );

		m_Biases.resize( m_NumFeatureMaps );
		m_DeltaBiases.resize( m_NumFeatureMaps );

		m_NetInputs.resize( m_NumOutputs );
		m_Activations.resize( m_NumOutputs );

		std::generate( m_Weights.begin(), m_Weights.end(), [&]() { return Random( -1.0f, 1.0f ); } );
		std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
	}

	virtual void Forward( const Tensor& _in, Tensor& _out ) override
	{

		uint32_t outSx = (m_InputSizeX - m_KernelSize + 1) / (m_Stride * m_Stride);
		uint32_t outSy = (m_InputSizeY - m_KernelSize + 1) / (m_Stride * m_Stride);

		_out.resize( m_NumOutputs );
		//std::fill( _out.begin(), _out.end(), 0.0f );
		
		float* accum = (float*)alloca( sizeof(float) * m_NumFeatureMaps );

		for( uint32_t y = 0 ; y < outSy ; ++y )
		{
			for( uint32_t x = 0 ; x < outSx ; ++x )
			{
				for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
				{
					accum[f] = m_Biases[f];
				}

				for( uint32_t ky = 0 ; ky < m_KernelSize ; ++ky )
				{
					uint32_t sy = y * m_Stride + ky;
					
					for( uint32_t kx = 0 ; kx < m_KernelSize ; ++kx )
					{
						uint32_t sx = x * m_Stride + kx;

						for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
						{
							accum[f] += _in[sy * m_InputSizeX + sx] * m_Weights[ f * m_KernelSize * m_KernelSize + ky * m_KernelSize + kx ];
						}
					}
				}
				
				for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
				{
					float output = Activation::Compute( accum[f]);
					uint32_t outIdx = f * outSx* outSy + y * outSx + x;
					
					_out[outIdx] = output;
				
					//TODO if( isTraining )
					{
						m_NetInputs[outIdx] = accum[f];
						m_Activations[outIdx] = output;
					}
				}
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
		for( uint32_t i = 0 ; i < m_DeltaWeights.size() ; ++i )
		{
			m_Weights[i] -= _learningRate * m_DeltaWeights[i];
		}

		for( uint32_t i = 0 ; i < m_DeltaBiases.size() ; ++i )
		{
			m_Biases[i] -= _learningRate * m_DeltaBiases[i];
		}		
	}

	virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
	{
		uint32_t outSx = (m_InputSizeX - m_KernelSize + 1) / (m_Stride * m_Stride);
		uint32_t outSy = (m_InputSizeY - m_KernelSize + 1) / (m_Stride * m_Stride);

		_inputGradients.resize( m_NumInputs, 0.0f );

		for( uint32_t y = 0 ; y < outSy ; ++y )
		{
			for( uint32_t x = 0 ; x < outSx ; ++x )
			{
				uint32_t outIdx = x + y * outSx;
				float dA_dN = Activation::ComputeDerivative( m_NetInputs[outIdx], m_Activations[outIdx] );
				float dE_dN = _outputGradients[outIdx] * dA_dN;

				for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
				{
					m_DeltaBiases[f] += dE_dN; //dN_dB is ignored because it is 1
				}

				for( uint32_t ky = 0 ; ky < m_KernelSize ; ++ky )
				{
					uint32_t sy = y * m_Stride + ky;

					for( uint32_t kx = 0 ; kx < m_KernelSize ; ++kx )
					{
						uint32_t sx = x * m_Stride + kx;

						for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
						{

							uint32_t weightIdx = f * m_KernelSize * m_KernelSize + ky * m_KernelSize + kx;

							float dN_dW = _layerInputs[sy * m_InputSizeX + sx];
							m_DeltaWeights[weightIdx] += dE_dN * dN_dW;

							_inputGradients[sy * m_InputSizeX + sx] += dE_dN * m_Weights[weightIdx];
						}
					}
				}
			}
		}

	}

	virtual const Tensor& GetOutput() const override { return m_Activations; }

private:
	uint32_t m_InputSizeX, m_InputSizeY, m_NumFeatureMaps, m_KernelSize, m_Stride;

	std::vector<float> m_Weights;
	std::vector<float> m_Biases;
	std::vector<float> m_DeltaWeights;
	std::vector<float> m_DeltaBiases;
	Tensor m_NetInputs;
	Tensor m_Activations;
};

