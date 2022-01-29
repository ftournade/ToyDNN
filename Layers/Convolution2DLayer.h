#pragma once

#include "Layer.h"

namespace ToyDNN
{

	class Convolution2D : public WeightsAndBiasesLayer
	{
	public:
		Convolution2D( uint32_t _numFeatureMaps=0, uint32_t _kernelSize=0, uint32_t _stride = 1 )
			: m_NumFeatureMaps( _numFeatureMaps ), m_KernelSize( _kernelSize ), m_Stride( _stride )
		{
		}

		virtual LayerType GetType() const { return LayerType::Convolution2D; }

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

			m_Output.resize( m_OutputShape.Size() );

			std::generate( m_Weights.begin(), m_Weights.end(), [&]() { return Random( -1.0f, 1.0f ); } );
			std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			_out.resize( m_OutputShape.Size() );
			//std::fill( _out.begin(), _out.end(), 0.0f );

			Scalar* accum = (Scalar*)alloca( sizeof( Scalar ) * m_OutputShape.m_SZ );

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
						uint32_t outIdx = m_OutputShape.Index( x, y, f );

						_out[outIdx] = accum[f];

						//TODO if( isTraining )
						{
							m_Output[outIdx] = accum[f];
						}
					}
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			_inputGradients.resize( m_InputShape.Size() );
			std::fill( _inputGradients.begin(), _inputGradients.end(), Scalar( 0.0 ) );

			Scalar* dE_dN = (Scalar*)alloca( sizeof( Scalar ) * m_OutputShape.m_SZ );

			for( uint32_t y = 0 ; y < m_OutputShape.m_SY ; ++y )
			{
				for( uint32_t x = 0 ; x < m_OutputShape.m_SX ; ++x )
				{
					for( uint32_t f = 0 ; f < m_OutputShape.m_SZ ; ++f )
					{
						uint32_t outIdx = m_OutputShape.Index( x, y, f );
						dE_dN[f] = _outputGradients[outIdx];

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

									Scalar dN_dW = _layerInputs[inIdx];
									m_DeltaWeights[weightIdx] += dE_dN[f] * dN_dW;

									_inputGradients[inIdx] += dE_dN[f] * m_Weights[weightIdx];
								}
							}
						}
					}
				}
			}

		}

		virtual const Tensor& GetOutput() const override { return m_Output; }

		virtual void Load( std::istream& _stream ) override
		{
			WeightsAndBiasesLayer::Load( _stream );

			_stream >> m_NumFeatureMaps >> m_KernelSize >> m_Stride >> m_KernelShape;
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			WeightsAndBiasesLayer::Save( _stream );

			_stream << m_NumFeatureMaps << m_KernelSize << m_Stride << m_KernelShape;
		}

	private:
		uint32_t m_NumFeatureMaps, m_KernelSize, m_Stride;
		TensorShape m_KernelShape;

		mutable Tensor m_Output; //needed for back prop
	};
}