#pragma once

#include "Layer.h"

namespace ToyDNN
{

	class MaxPooling : public Layer
	{
	public:
		MaxPooling( uint32_t _poolSizeX, uint32_t _poolSizeY ) :
			m_PoolSizeX(_poolSizeX), m_PoolSizeY(_poolSizeY)
		{
			assert( (_poolSizeX < 16) && (_poolSizeY < 16) ); //PixelCoord is 4 bit encoded
		}

		virtual void Setup( const TensorShape& _previousLayerOutputShape ) override
		{
			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = TensorShape( _previousLayerOutputShape.m_SX / m_PoolSizeX, _previousLayerOutputShape.m_SY / m_PoolSizeY, _previousLayerOutputShape.m_SZ );

			m_Output.resize( m_OutputShape.Size() );
			m_MaxElement.resize( m_OutputShape.Size() );
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			_out.resize( m_OutputShape.Size() );

			for( int z = 0 ; z < (int)m_OutputShape.m_SZ ; ++z )
			{
				for( int y = 0 ; y < (int)m_OutputShape.m_SY ; ++y )
				{
					for( int x = 0 ; x < (int)m_OutputShape.m_SX ; ++x )
					{
						uint32_t outIdx = m_OutputShape.Index( x, y, z );
						
						//TODO handle out of bound

						float maxValue = -FLT_MAX;
						PixelCoord maxElemCoord;

						for( uint32_t py = 0 ; py < m_PoolSizeY ; ++py )
						{
							for( uint32_t px = 0 ; px < m_PoolSizeX ; ++px )
							{
								uint32_t inIdx = m_InputShape.Index( x * m_PoolSizeX + px, y * m_PoolSizeY + py, z );
								float v = _in[inIdx];

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

		virtual void ClearWeightDeltas() override {}
		virtual void ApplyWeightDeltas( float _learningRate ) override {}

		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
			_inputGradients.resize( m_InputShape.Size(), 0.0f );

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

						_inputGradients[inIdx] = 1.0f;
					}
				}
			}
		}

		virtual const Tensor& GetOutput() const override { return m_Output; }

	private:
		uint32_t m_PoolSizeX, m_PoolSizeY;
		mutable Tensor m_Output;

		struct PixelCoord
		{
			uint8_t x : 4;
			uint8_t y : 4;
		};

		mutable std::vector<PixelCoord> m_MaxElement; //used by back propagation
	};
}
