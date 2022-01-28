#pragma once

#include "Tensor.h"
#include "Util.h"

#undef min
#undef max
#include <algorithm>
#include <assert.h>

namespace ToyDNN
{

	class Layer
	{
	public:
		Layer() {}
		virtual ~Layer() {}
		virtual void Setup( const TensorShape& _previousLayerOutputShape ) = 0;
		virtual void Forward( const Tensor& _in, Tensor& _out ) const = 0;
		virtual void ClearWeightDeltas() = 0;
		virtual void ApplyWeightDeltas( Scalar _learningRate ) = 0;
		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients/*in*/, Tensor& _inputGradients /*out*/ ) = 0;
		virtual bool GetRandomParameterAndAssociatedGradient( Scalar** _parameter, Scalar& _gradient ) { return false; } //used for gradient checking
		virtual const Tensor& GetOutput() const = 0;

		inline const TensorShape& GetInputShape() const { return m_InputShape; }
		inline const TensorShape& GetOutputShape() const { return m_OutputShape; }

	protected:
		TensorShape m_InputShape, m_OutputShape;
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

	protected:
		std::vector<Scalar> m_Weights;
		std::vector<Scalar> m_Biases;
		std::vector<Scalar> m_DeltaWeights;
		std::vector<Scalar> m_DeltaBiases;
	};

}