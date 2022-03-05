#pragma once

#include "Layer.h"

namespace ToyDNN
{

	class MaxPooling : public Layer
	{
	public:
		MaxPooling( uint32_t _poolSizeX=2, uint32_t _poolSizeY=2 ) :
			m_PoolSizeX(_poolSizeX), m_PoolSizeY(_poolSizeY)
		{
			assert( (_poolSizeX < 16) && (_poolSizeY < 16) ); //PixelCoord is 4 bit encoded
		}

		virtual LayerType GetType() const override { return LayerType::MaxPooling; }
		virtual const char* GetName() const override { return "MaxPooling"; }

		virtual void Setup( const TensorShape& _previousLayerOutputShape, uint32_t _outputPadding ) override
		{
			//TODO support padding
			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = TensorShape( (_previousLayerOutputShape.m_SX + m_PoolSizeX  - 1) / m_PoolSizeX, 
										 (_previousLayerOutputShape.m_SY + m_PoolSizeY  - 1) / m_PoolSizeY, 
										 _previousLayerOutputShape.m_SZ );

			m_MaxElement.resize( m_OutputShape.Size() );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			//#pragma omp parallel for
			for( int z = 0 ; z < (int)m_OutputShape.m_SZ ; ++z )
			{
				for( int y = 0 ; y < (int)m_OutputShape.m_SY ; ++y )
				{
					for( int x = 0 ; x < (int)m_OutputShape.m_SX ; ++x )
					{
						uint32_t outIdx = m_OutputShape.Index( x, y, z );
						
						Scalar maxValue = sizeof(Scalar) == sizeof(double) ? -DBL_MAX : -FLT_MAX;
						PixelCoord maxElemCoord = { 0,0 };

						uint32_t poolSizeY = std::min( m_PoolSizeY, m_InputShape.m_SY - y * m_PoolSizeY );

						for( uint32_t py = 0 ; py < poolSizeY ; ++py )
						{
							uint32_t poolSizeX = std::min( m_PoolSizeX, m_InputShape.m_SX - x * m_PoolSizeX );

							for( uint32_t px = 0 ; px < poolSizeX ; ++px )
							{
								uint32_t inIdx = m_InputShape.Index( x * m_PoolSizeX + px, y * m_PoolSizeY + py, z );
								Scalar v = _in[inIdx];

								//TODO if( IsTraining )
								if( v > maxValue )
								{
									maxValue = v;
									maxElemCoord.x = px;
									maxElemCoord.y = py;
								}
								//else maxValue = std::max( maxValue, v );
							}
						}

						_out[outIdx] = maxValue;
						m_MaxElement[outIdx] = maxElemCoord;
					}
				}
			}
		}

		
		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			std::fill( _inputGradients.begin(), _inputGradients.end(), Scalar( 0.0 ) );
				
			//#pragma omp parallel for
			for( int z = 0 ; z < (int)m_OutputShape.m_SZ ; ++z )
			{
				for( int y = 0 ; y < (int)m_OutputShape.m_SY ; ++y )
				{
					for( int x = 0 ; x < (int)m_OutputShape.m_SX ; ++x )
					{
						uint32_t outIdx = m_OutputShape.Index( x, y, z );

						const PixelCoord p = m_MaxElement[outIdx];
						
						uint32_t xin = x * m_PoolSizeX + p.x;
						uint32_t yin = y * m_PoolSizeY + p.y;
						
						uint32_t inIdx = m_InputShape.Index( xin, yin, z );

						_inputGradients[inIdx] = _outputGradients[outIdx];
					}
				}
			}
		}

		virtual void Load( std::istream& _stream ) override
		{
			Layer::Load( _stream );

			Read( _stream, m_PoolSizeX );
			Read( _stream, m_PoolSizeY );

			m_MaxElement.resize( m_OutputShape.Size() ); //TODO don't do this if only inferring
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			Layer::Save( _stream );

			Write( _stream, m_PoolSizeX );
			Write( _stream, m_PoolSizeY );
		}

	private:
		uint32_t m_PoolSizeX, m_PoolSizeY;

		struct PixelCoord
		{
			uint8_t x : 4;
			uint8_t y : 4;
		};

		mutable std::vector<PixelCoord> m_MaxElement; //used by back propagation
	};
}
