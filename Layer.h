#pragma once

#include "Tensor.h"
#include "Util.h"

#undef min
#undef max
#include <algorithm>
#include <assert.h>
#include <cmath>

namespace ToyDNN
{

	enum class LayerType : uint16_t
	{
		FullyConnected,
		Convolution2D,
		MaxPooling,

		Relu,
		LeakyRelu,
		Sigmoid,
		Tanh,
		SoftMax
	};

	class Layer
	{
	public:
		Layer() {}
		virtual ~Layer() {}

		virtual LayerType GetType() const = 0;
		virtual void Setup( const TensorShape& _previousLayerOutputShape ) = 0;
		virtual void Forward( const Tensor& _in, Tensor& _out ) const = 0;
		virtual void ClearWeightDeltas() = 0;
		virtual void ApplyWeightDeltas( Scalar _learningRate ) = 0;
		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients/*in*/, Tensor& _inputGradients /*out*/ ) = 0;
		virtual bool GetRandomParameterAndAssociatedGradient( Scalar** _parameter, Scalar& _gradient ) { return false; } //used for gradient checking

		inline const TensorShape& GetInputShape() const { return m_InputShape; }
		inline const TensorShape& GetOutputShape() const { return m_OutputShape; }
		inline const Tensor& GetOutput() const { return m_Output; }

		void CacheOutput( const Tensor& _output ) const { m_Output = _output; }

		virtual void Load( std::istream& _stream ) 
		{
			Read( _stream, m_InputShape );
			Read( _stream, m_OutputShape );
		}

		virtual void Save( std::ostream& _stream ) const
		{
			Write( _stream, m_InputShape );
			Write( _stream, m_OutputShape );
		}

	protected:
		TensorShape m_InputShape, m_OutputShape;

	private:
		mutable Tensor m_Output; //Only used during training for back propagation
	};

	class WeightsAndBiasesLayer : public Layer
	{
	public:
		virtual void ClearWeightDeltas() override
		{
			std::fill( m_DeltaWeights.begin(), m_DeltaWeights.end(), 0.0f );
			std::fill( m_DeltaBiases.begin(), m_DeltaBiases.end(), 0.0f );
		}

		virtual void ApplyWeightDeltas( Scalar _learningRate ) override
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

		virtual bool GetRandomParameterAndAssociatedGradient( Scalar** _parameter, Scalar& _gradient ) override
		{
			if( m_Weights.empty() || m_Biases.empty() )
				return false;

			if( rand() % 2 == 0 )
			{
				uint32_t weightIdx = rand() % m_Weights.size();
				*_parameter = &m_Weights[weightIdx];
				_gradient = m_DeltaWeights[weightIdx];
			}
			else
			{
				uint32_t biasIdx = rand() % m_Biases.size();
				*_parameter = &m_Biases[biasIdx];
				_gradient = m_DeltaBiases[biasIdx];
			}

			return true;
		}

		virtual void Load( std::istream& _stream ) override
		{
			Layer::Load( _stream );
			
			size_t numWeights;
			Read( _stream, numWeights );
			m_Weights.resize( numWeights );
			_stream.read( (char*)&m_Weights[0], m_Weights.size() * sizeof( m_Weights[0] ) );

			size_t numBiases;
			Read( _stream, numBiases );
			m_Biases.resize( numBiases );
			_stream.read( (char*)&m_Biases[0], m_Biases.size() * sizeof( m_Biases[0] ) );
		}

		virtual void Save( std::ostream& _stream ) const override
		{
			Layer::Save( _stream );
			
			Write( _stream, m_Weights.size() );
			_stream.write( (const char*)&m_Weights[0], m_Weights.size() * sizeof( m_Weights[0] ) );

			Write( _stream, m_Biases.size() );
			_stream.write( (const char*)&m_Biases[0], m_Biases.size() * sizeof( m_Biases[0] ) );
		}

	protected:
		std::vector<Scalar> m_Weights;
		std::vector<Scalar> m_Biases;
		std::vector<Scalar> m_DeltaWeights;
		std::vector<Scalar> m_DeltaBiases;
	};

	Layer* CreateLayer( LayerType _type );
}