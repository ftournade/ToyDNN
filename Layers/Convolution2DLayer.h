#pragma once

#include "Layer.h"

template <typename Activation>
class Convolution2DLayer : public Layer
{
public:
	Convolution2DLayer( uint32_t _numFeatureMaps, uint32_t _kernelSize, uint32_t _stride = 1 )
		: m_NumFeatureMaps( _numFeatureMaps ), m_KernelSize( _kernelSize ), m_Stride( _stride )
	{
	}

	virtual void Setup( const TensorShape& _previousLayerOutputShape ) override
	{
		m_InputShape = _previousLayerOutputShape;
		m_OutputShape = TensorShape( (m_InputShape.m_SX - m_KernelSize + 1) / m_Stride,
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

	virtual void Forward( const Tensor& _in, Tensor& _out ) const override
	{
		_out.resize( m_OutputShape.Size() );
		//std::fill( _out.begin(), _out.end(), 0.0f );

		float* accum = (float*)alloca( sizeof( float ) * m_OutputShape.m_SZ );

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
	mutable Tensor m_NetInputs;
	mutable Tensor m_Activations;
};
