#pragma once

#include "Layer.h"

namespace ToyDNN
{

	
	class PaddingLayer : public Layer
	{
	private:
		//Layer is not meant to be declared explicitely by user, rather it is implicitely added before the first layer 
		//by NeuralNetwork::Compile when the first layer needs input padding, or by CreateLayer during network loading
		friend class NeuralNetwork;
		friend Layer* CreateLayer( LayerType _type );

		PaddingLayer() :
			m_Padding( 0 )
		{
		}

	public:
		virtual LayerType GetType() const override { return LayerType::Padding; }
		virtual const char* GetName() const override { return "Padding"; }

		virtual void Setup( const TensorShape& _previousLayerOutputShape, uint32_t _outputPadding ) override
		{
			m_Padding = _outputPadding;
			m_InputShape = _previousLayerOutputShape;
			m_OutputShape = _previousLayerOutputShape;
			m_OutputShape.m_Padding = m_Padding;
			m_OutputShape.m_SX += m_Padding;
			m_OutputShape.m_SY += m_Padding;
		}

		virtual void Forward( const Tensor& _in, Tensor& _out ) const override
		{
			memset( &_out[0], 0, _out.size() * sizeof( Scalar ) );

			for( uint32_t z = 0 ; z < m_InputShape.m_SZ ; ++z )
			{
				for( uint32_t y = 0 ; y < m_InputShape.m_SY ; ++y )
				{
					for( uint32_t x = 0 ; x < m_InputShape.m_SX ; ++x )
					{
						_out[m_OutputShape.PaddedIndex( x, y, z )] = _in[m_InputShape.Index( x, y, z )];
					}
				}
			}
		}


		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients, Tensor& _inputGradients ) override
		{
		}

		virtual void Load( std::istream& _stream ) override
		{
			Layer::Load( _stream );

			Read( _stream, m_Padding );
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			Layer::Save( _stream );

			Write( _stream, m_Padding );
		}

	private:
		uint32_t m_Padding;
	};
}
