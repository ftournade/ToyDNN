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
		virtual void ApplyWeightDeltas( float _learningRate ) = 0;
		virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients/*in*/, Tensor& _inputGradients /*out*/ ) = 0;
		virtual const Tensor& GetOutput() const = 0;

		inline const TensorShape& GetInputShape() const { return m_InputShape; }
		inline const TensorShape& GetOutputShape() const { return m_OutputShape; }
	protected:
		TensorShape m_InputShape, m_OutputShape;

	};

}