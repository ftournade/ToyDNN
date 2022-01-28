#pragma once

#include <vector>

namespace ToyDNN
{
	typedef std::vector< float > Tensor;

	class TensorShape
	{
	public:
		inline TensorShape() : m_SX( 0 ), m_SY( 0 ), m_SZ( 0 ) {}
		inline TensorShape( uint32_t _sx, uint32_t _sy = 1, uint32_t _sz = 1 ) : m_SX( _sx ), m_SY( _sy ), m_SZ( _sz ) {}
		inline uint32_t Size() const { return m_SX * m_SY * m_SZ; }
		inline uint32_t Index( uint32_t _x, uint32_t _y, uint32_t _z ) const { return m_SX * (m_SY * _z + _y) + _x; }

		uint32_t m_SX, m_SY, m_SZ;
	};
}
