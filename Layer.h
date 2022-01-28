#pragma once

#include "Tensor.h"
#include "ActivationFuncs.h"

#include <algorithm>
#include <assert.h>


inline float Lerp( float t, float a, float b )
{
	return a + t * (b - a);
}

inline float Random( float _min, float _max )
{
	float u = rand() / (float)RAND_MAX;
	return Lerp( u, _min, _max );
}

class TensorShape
{
public:
	inline TensorShape() : m_SX( 0 ), m_SY( 0 ), m_SZ( 0 ) {}
	inline TensorShape( uint32_t _sx, uint32_t _sy=1, uint32_t _sz=1 ) : m_SX(_sx), m_SY(_sy), m_SZ(_sz) {}
	inline uint32_t Size() const { return m_SX * m_SY * m_SZ; }
	inline uint32_t Index( uint32_t _x, uint32_t _y, uint32_t _z ) const { return m_SX * ( m_SY * _z + _y ) + _x; }

	uint32_t m_SX, m_SY, m_SZ;
};

class Layer
{
public:
	Layer() {}
	virtual ~Layer() {}
	virtual void Setup( const TensorShape& _previousLayerOutputShape ) = 0;
	virtual void Forward( const Tensor& _in, Tensor& _out ) const = 0;
	virtual void ClearWeightDeltas() = 0;
	virtual void ApplyWeightDeltas( float _learningRate ) = 0;
	virtual void BackPropagation( const Tensor& _layerInputs, const Tensor& _outputGradients/*in*/, Tensor& _inputGradients /*out*/) = 0;
	virtual const Tensor& GetOutput() const = 0;

	inline const TensorShape& GetInputShape() const { return m_InputShape; }
	inline const TensorShape& GetOutputShape() const { return m_OutputShape; }
protected:
	TensorShape m_InputShape, m_OutputShape;

};

