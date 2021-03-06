#pragma once

#include "Layer.h"

namespace ToyDNN
{
	enum class Padding
	{
		Valid,
		Same
	};

	class Convolution2D : public WeightsAndBiasesLayer
	{
	public:
		Convolution2D( uint32_t _numFeatureMaps=0, uint32_t _kernelSize=1, uint32_t _stride = 1, Padding _padding = Padding::Same )
			: m_NumFeatureMaps( _numFeatureMaps ), m_KernelSize( _kernelSize ), m_Stride( _stride ), m_Padding( _padding )
		{
			assert( m_KernelSize % 2 == 1 ); //TODO support even kernels ?
		}

		virtual LayerType GetType() const override { return LayerType::Convolution2D; }
		virtual const char* GetName() const override { return "Convolution2D"; }

		virtual uint32_t GetInputPadding() const override
		{
			assert( m_KernelSize % 2 == 1 ); //TODO support even kernels ?

			if( m_Padding == Padding::Same )
			{
				return m_KernelSize - m_Stride;
			}
			else
			{
				return 0;
			}
		}

		virtual void Setup( const TensorShape& _previousLayerOutputShape, uint32_t _outputPadding ) override
		{
			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = TensorShape( _outputPadding + (m_InputShape.m_SX - m_KernelSize + m_Stride ) / m_Stride,
										 _outputPadding + (m_InputShape.m_SY - m_KernelSize + m_Stride ) / m_Stride,
										 m_NumFeatureMaps,
										 _outputPadding );

			m_KernelShape = TensorShape( m_KernelSize, m_KernelSize, m_InputShape.m_SZ );
			m_Weights.resize( m_NumFeatureMaps * m_KernelShape.Size() );
			m_WeightGradients.resize( m_Weights.size() );

			m_Biases.resize( m_NumFeatureMaps );
			m_BiasGradients.resize( m_NumFeatureMaps );

			uint32_t fanIn = m_KernelSize * m_KernelSize * m_InputShape.m_SZ;
			uint32_t fanOut = (m_KernelSize / m_Stride) * (m_KernelSize / m_Stride) * m_NumFeatureMaps;

			WeightInit::He( fanIn, fanOut, m_Weights );
			std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			
			Scalar* accum = (Scalar*)alloca( sizeof( Scalar ) * m_OutputShape.m_SZ );

			for( uint32_t y = 0 ; y < m_OutputShape.m_SY - m_OutputShape.m_Padding ; ++y )
			{
				for( uint32_t x = 0 ; x < m_OutputShape.m_SX - m_OutputShape.m_Padding ; ++x )
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
						uint32_t outIdx = m_OutputShape.PaddedIndex( x, y, f );

						_out[outIdx] = accum[f];
					}
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			std::fill( _inputGradients.begin(), _inputGradients.end(), Scalar( 0.0 ) );

			Scalar* dE_dN = (Scalar*)alloca( sizeof( Scalar ) * m_OutputShape.m_SZ );

			for( uint32_t y = 0 ; y < m_OutputShape.m_SY - m_OutputShape.m_Padding ; ++y )
			{
				for( uint32_t x = 0 ; x < m_OutputShape.m_SX - m_OutputShape.m_Padding ; ++x )
				{
					for( uint32_t f = 0 ; f < m_OutputShape.m_SZ ; ++f )
					{
						uint32_t outIdx = m_OutputShape.PaddedIndex( x, y, f );
						dE_dN[f] = _outputGradients[outIdx];

						m_BiasGradients[f] += dE_dN[f]; //dN_dB is ignored because it is 1
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
									m_WeightGradients[weightIdx] += dE_dN[f] * dN_dW;

									_inputGradients[inIdx] += dE_dN[f] * m_Weights[weightIdx];
								}
							}
						}
					}
				}
			}

		}

		virtual void Load( std::istream& _stream ) override
		{
			WeightsAndBiasesLayer::Load( _stream );

			Read( _stream, m_NumFeatureMaps );
			Read( _stream, m_KernelSize );
			Read( _stream, m_Stride );
			Read( _stream, m_KernelShape );
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			WeightsAndBiasesLayer::Save( _stream );

			Write( _stream, m_NumFeatureMaps );
			Write( _stream, m_KernelSize );
			Write( _stream, m_Stride );
			Write( _stream, m_KernelShape );
		}

		inline uint32_t GetKernelSize() const { return m_KernelSize; }
		inline uint32_t GetNumFeatureMaps() const { return m_NumFeatureMaps; }

	private:
		uint32_t m_NumFeatureMaps, m_KernelSize, m_Stride;
		Padding m_Padding;
		TensorShape m_KernelShape;
	};


	//===========================================================


	class ConvolutionTranspose2D : public WeightsAndBiasesLayer
	{
	public:
		ConvolutionTranspose2D( uint32_t _numFeatureMaps = 0, uint32_t _kernelSize = 0, uint32_t _stride = 1, Padding _padding = Padding::Same )
			: m_NumFeatureMaps( _numFeatureMaps ), m_KernelSize( _kernelSize ), m_Stride( _stride ), m_Padding( _padding )
		{
		}

		virtual LayerType GetType() const override { return LayerType::ConvolutionTranspose2D; }
		virtual const char* GetName() const override { return "ConvolutionTranspose2D"; }

		virtual void Setup( const TensorShape& _previousLayerOutputShape, uint32_t _outputPadding ) override
		{
			uint32_t padding;

			if( m_Padding == Padding::Same )
			{
				padding = m_KernelSize - m_Stride;
			}
			else
			{
				assert( false ); //codepath untested
				padding = 0;
			}


			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = TensorShape( (m_InputShape.m_SX - m_InputShape.m_Padding - 1) * m_Stride + m_KernelSize,
										 (m_InputShape.m_SY - m_InputShape.m_Padding - 1) * m_Stride + m_KernelSize,
										 m_NumFeatureMaps,
										 std::max( padding, _outputPadding ) );

			m_KernelShape = TensorShape( m_KernelSize, m_KernelSize, m_InputShape.m_SZ );
			m_Weights.resize( m_NumFeatureMaps * m_KernelShape.Size() );
			m_WeightGradients.resize( m_Weights.size() );

			m_Biases.resize( m_NumFeatureMaps );
			m_BiasGradients.resize( m_NumFeatureMaps );

			//TODO probably incorrect
			uint32_t fanIn = m_KernelSize * m_KernelSize * m_InputShape.m_SZ;
			uint32_t fanOut = (m_KernelSize * m_Stride) * (m_KernelSize * m_Stride) * m_NumFeatureMaps;

			WeightInit::He( fanIn, fanOut, m_Weights );
			std::fill( m_Biases.begin(), m_Biases.end(), 0.0f );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			std::fill( _out.begin(), _out.end(), Scalar( 0.0 ) );//TODO memset ?

			for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
			{
				for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
				{					
					for( uint32_t kz = 0 ; kz < m_InputShape.m_SZ ; ++kz )
					{
						uint32_t inIdx = m_InputShape.PaddedIndex( x, y, kz );
						Scalar in = _in[inIdx];

						for( uint32_t ky = 0 ; ky < m_KernelSize ; ++ky )
						{
							uint32_t sy = y * m_Stride + ky;

							for( uint32_t kx = 0 ; kx < m_KernelSize ; ++kx )
							{
								uint32_t sx = x * m_Stride + kx;

								for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
								{
									uint32_t outIdx = m_OutputShape.Index( sx, sy, f );
									uint32_t weightIdx = m_KernelShape.Size() * f + m_KernelShape.Index( kx, ky, kz );
									_out[outIdx] += m_Biases[f] + m_Weights[weightIdx] * in;
									//TODO test average instead of add to avoid checkerboard artefacts
								}
							}
						}
					}
				}
			}
		}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			std::fill( _inputGradients.begin(), _inputGradients.end(), Scalar( 0.0 ) );

			Scalar* dE_dN = (Scalar*)alloca( sizeof( Scalar ) * m_OutputShape.m_SZ );
			memset( dE_dN, 0, sizeof( Scalar ) * m_OutputShape.m_SZ );

			for( uint32_t y = 0 ; y < m_InputShape.m_SY - m_InputShape.m_Padding ; ++y )
			{
				for( uint32_t x = 0 ; x < m_InputShape.m_SX - m_InputShape.m_Padding ; ++x )
				{
					for( uint32_t kz = 0 ; kz < m_InputShape.m_SZ ; ++kz )
					{
						uint32_t inIdx = m_InputShape.PaddedIndex( x, y, kz );
						Scalar in = _layerInputs[inIdx];

						for( uint32_t ky = 0 ; ky < m_KernelSize ; ++ky )
						{
							uint32_t sy = y * m_Stride + ky;

							for( uint32_t kx = 0 ; kx < m_KernelSize ; ++kx )
							{
								uint32_t sx = x * m_Stride + kx;

								for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
								{
									uint32_t outIdx = m_OutputShape.Index( sx, sy, f );
									uint32_t weightIdx = m_KernelShape.Size() * f + m_KernelShape.Index( kx, ky, kz );
									
									dE_dN[f] += _outputGradients[outIdx];

									m_BiasGradients[f] += dE_dN[f];
									Scalar dN_dW = in;
									m_WeightGradients[weightIdx] += dE_dN[f] * dN_dW;
								}
							}
						}
					}
				}
			}
		}

		virtual void Load( std::istream& _stream ) override
		{
			WeightsAndBiasesLayer::Load( _stream );

			Read( _stream, m_NumFeatureMaps );
			Read( _stream, m_KernelSize );
			Read( _stream, m_Stride );
			Read( _stream, m_KernelShape );
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			WeightsAndBiasesLayer::Save( _stream );

			Write( _stream, m_NumFeatureMaps );
			Write( _stream, m_KernelSize );
			Write( _stream, m_Stride );
			Write( _stream, m_KernelShape );
		}

		inline uint32_t GetKernelSize() const { return m_KernelSize; }
		inline uint32_t GetNumFeatureMaps() const { return m_NumFeatureMaps; }

	private:
		uint32_t m_NumFeatureMaps, m_KernelSize, m_Stride;
		Padding m_Padding;
		TensorShape m_KernelShape;
	};

}