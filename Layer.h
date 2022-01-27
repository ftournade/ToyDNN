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

class TensorShape
{
public:
	inline TensorShape() : m_SX( 0 ), m_SY( 0 ), m_SZ( 0 ) {}
	inline TensorShape( uint32_t _sx, uint32_t _sy=1, uint32_t _sz=1 ) : m_SX(_sx), m_SY(_sy), m_SZ(_sz) {}
	inline uint32_t Size() const { return m_SX * m_SY * m_SZ; }
	inline uint32_t Index( uint32_t _x, uint32_t _y, uint32_t _z ) const { return m_SX * ( m_SY * _z + _y ) + _x; }

	uint32_t m_SX, m_SY, m_SZ;
};

class Layer
{
public:
	Layer() {}
	virtual ~Layer() {}
	virtual void Setup( const TensorShape& _previousLayerOutputShape ) = 0;
	virtual void Forward( const Tensor& _in, Tensor& _out ) = 0;
	virtual void ClearWeightDeltas() = 0;
	virtual void ApplyWeightDeltas( float _learningRate ) = 0;
	virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients/*in*/, Tensor& _inputGradients /*out*/) = 0;
	virtual const Tensor& GetOutput() const = 0;

	inline const TensorShape& GetInputShape() const { return m_InputShape; }
	inline const TensorShape& GetOutputShape() const { return m_OutputShape; }
protected:
	TensorShape m_InputShape, m_OutputShape;

};

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

	virtual void Forward( const Tensor& _in, Tensor& _out ) override
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
	Tensor m_NetInputs;
	Tensor m_Activations;
};

template <typename Activation>
class Convolution2DLayer : public Layer
{
public:
	Convolution2DLayer( uint32_t _numFeatureMaps, uint32_t _kernelSize, uint32_t _stride=1 ) 
	: m_NumFeatureMaps(_numFeatureMaps), m_KernelSize(_kernelSize), m_Stride(_stride) 
	{}

	virtual void Setup( const TensorShape& _previousLayerOutputShape ) override
	{
		m_InputShape = _previousLayerOutputShape;
		m_OutputShape = TensorShape(	(m_InputShape.m_SX - m_KernelSize + 1) / m_Stride,
										(m_InputShape.m_SY - m_KernelSize + 1) / m_Stride,
										m_NumFeatureMaps );
		
		m_KernelShape = TensorShape( m_KernelSize, m_KernelSize, m_InputShape.m_SZ );
		m_Weights.resize( m_NumFeatureMaps * m_KernelShape.Size() );
		m_DeltaWeights.resize( m_Weights.size() );

		m_Biases.resize( m_NumFeatureMaps );
		m_DeltaBiases.resize( m_NumFeatureMaps );

		m_NetInputs.resize( m_OutputShape.Size() );
		m_Activations.resize( m_OutputShape.Size() );

		std::generate( m_Weights.begin(), m_Weights.end(), [&]() { return Random( -1.0f, 1.0f ); } );
		std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
	}

	virtual void Forward( const Tensor& _in, Tensor& _out ) override
	{
		_out.resize( m_OutputShape.Size() );
		//std::fill( _out.begin(), _out.end(), 0.0f );
		
		float* accum = (float*)alloca( sizeof(float) * m_OutputShape.m_SZ );

		for( uint32_t y = 0 ; y < m_OutputShape.m_SY ; ++y )
		{
			for( uint32_t x = 0 ; x < m_OutputShape.m_SX ; ++x )
			{
				for( uint32_t f = 0 ; f < m_OutputShape.m_SZ ; ++f )
				{
					accum[f] = m_Biases[f];
				}

				for( uint32_t kz = 0 ; kz < m_InputShape.m_SZ ; ++kz )
				{
					for( uint32_t ky = 0 ; ky < m_KernelSize ; ++ky )
					{
						uint32_t sy = y * m_Stride + ky;

						for( uint32_t kx = 0 ; kx < m_KernelSize ; ++kx )
						{
							uint32_t sx = x * m_Stride + kx;

							for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
							{
								uint32_t inIdx = m_InputShape.Index( sx, sy, kz );
								uint32_t weightIdx = m_KernelShape.Size() * f + m_KernelShape.Index( kx, ky, kz );

								accum[f] += _in[inIdx] * m_Weights[weightIdx];
							}
						}
					}
				}

				for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
				{
					float output = Activation::Compute( accum[f] );
					uint32_t outIdx = m_OutputShape.Index( x, y, f );
					
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

		_inputGradients.resize( m_InputShape.Size(), 0.0f );

		float* dE_dN = (float*)alloca( sizeof( float ) * m_OutputShape.m_SZ );

		for( uint32_t y = 0 ; y < m_OutputShape.m_SY ; ++y )
		{
			for( uint32_t x = 0 ; x < m_OutputShape.m_SX ; ++x )
			{
				for( uint32_t f = 0 ; f < m_OutputShape.m_SZ ; ++f )
				{
					uint32_t outIdx = m_OutputShape.Index( x, y, f );
					float dA_dN = Activation::ComputeDerivative( m_NetInputs[outIdx], m_Activations[outIdx] );
					dE_dN[f] = _outputGradients[outIdx] * dA_dN;

					m_DeltaBiases[f] += dE_dN[f]; //dN_dB is ignored because it is 1
				}

				for( uint32_t kz = 0 ; kz < m_InputShape.m_SZ ; ++kz )
				{
					for( uint32_t ky = 0 ; ky < m_KernelSize ; ++ky )
					{
						uint32_t sy = y * m_Stride + ky;

						for( uint32_t kx = 0 ; kx < m_KernelSize ; ++kx )
						{
							uint32_t sx = x * m_Stride + kx;

							for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
							{
								uint32_t inIdx = m_InputShape.Index( sx, sy, kz );
								uint32_t weightIdx = m_KernelShape.Size() * f + m_KernelShape.Index( kx, ky, kz );

								float dN_dW = _layerInputs[inIdx];
								m_DeltaWeights[weightIdx] += dE_dN[f] * dN_dW;

								_inputGradients[inIdx] += dE_dN[f] * m_Weights[weightIdx];
							}
						}
					}
				}
			}
		}

	}

	virtual const Tensor& GetOutput() const override { return m_Activations; }

private:
	uint32_t m_NumFeatureMaps, m_KernelSize, m_Stride;
	TensorShape m_KernelShape;

	std::vector<float> m_Weights;
	std::vector<float> m_Biases;
	std::vector<float> m_DeltaWeights;
	std::vector<float> m_DeltaBiases;
	Tensor m_NetInputs;
	Tensor m_Activations;
};
