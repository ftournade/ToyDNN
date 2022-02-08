#pragma once

#include "Tensor.h"
#include "Optimizers.h"
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
		virtual void ClearGradients() {}
		virtual void ScaleGradients( Scalar _scale ) {}
		virtual void ApplyGradients( Optimizer& _optimizer ) {}
		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _output, Tensor& _inputGradients ) = 0;
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
		virtual void ClearGradients() override
		{
			std::fill( m_WeightGradients.begin(), m_WeightGradients.end(), 0.0f );
			std::fill( m_BiasGradients.begin(), m_BiasGradients.end(), 0.0f );
		}

		virtual void ScaleGradients( Scalar _scale ) override
		{
			auto scaler = [&]( Scalar s )->Scalar { return s * _scale; };

			std::transform( m_WeightGradients.begin(), m_WeightGradients.end(), m_WeightGradients.begin(), scaler );
			std::transform( m_BiasGradients.begin(), m_BiasGradients.end(), m_BiasGradients.begin(), scaler );
		}

		virtual void ApplyGradients( Optimizer& _optimizer ) override
		{
			_optimizer.UpdateTrainableParameters( m_WeightGradients, m_Weights );
			_optimizer.UpdateTrainableParameters( m_BiasGradients, m_Biases );
		}

		virtual bool GetRandomParameterAndAssociatedGradient( Scalar** _parameter, Scalar& _gradient ) override
		{
			if( m_Weights.empty() || m_Biases.empty() )
				return false;

			if( rand() % 2 == 0 )
			{
				uint32_t weightIdx = rand() % m_Weights.size();
				*_parameter = &m_Weights[weightIdx];
				_gradient = m_WeightGradients[weightIdx];
			}
			else
			{
				uint32_t biasIdx = rand() % m_Biases.size();
				*_parameter = &m_Biases[biasIdx];
				_gradient = m_BiasGradients[biasIdx];
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
		std::vector<Scalar> m_WeightGradients;
		std::vector<Scalar> m_BiasGradients;
	};

	Layer* CreateLayer( LayerType _type );
}