#pragma once

#include <vector>
#include <cmath>
#include <assert.h>

//#define CHECK_NAN_AND_INF

#ifdef CHECK_NAN_AND_INF
	#define ASSERT_IS_FINITE( s ) if( !std::isfinite(s) ) { __debugbreak();}
#else
	#define ASSERT_IS_FINITE( s )
#endif

namespace ToyDNN
{
	typedef double Scalar;
	typedef std::vector< Scalar > Tensor;

	inline void AssertIsFinite( const std::vector< Scalar >& _scalars )
	{
		#ifdef CHECK_NAN_AND_INF
		for( Scalar s : _scalars )
		{
			ASSERT_IS_FINITE( s )
		}
		#endif		
	}

	class TensorShape
	{
	public:
		inline TensorShape() : m_SX( 0 ), m_SY( 0 ), m_SZ( 0 ), m_Padding( 0 ) {}
		inline TensorShape( uint32_t _sx, uint32_t _sy = 1, uint32_t _sz = 1, uint32_t _padding = 0 ) : 
			m_SX( _sx ), m_SY( _sy ), m_SZ( _sz ), 
			m_Padding( _padding ) 
		{}
		
		inline uint32_t Size() const { return m_SX * m_SY * m_SZ; }
		inline uint32_t SizeWithoutPadding() const { return (m_SX - m_Padding) * (m_SY - m_Padding) * m_SZ; }

		inline uint32_t Index( uint32_t _x, uint32_t _y, uint32_t _z ) const 
		{
			assert( _x < m_SX );
			assert( _y < m_SY );
			assert( _z < m_SZ );
			return m_SX * (m_SY * _z + _y) + _x; 
		}

		inline uint32_t PaddedIndex( uint32_t _x, uint32_t _y, uint32_t _z ) const
		{
			_x += m_Padding / 2;
			_y += m_Padding / 2;
			assert( _x < m_SX );
			assert( _y < m_SY );
			assert( _z < m_SZ );
			return m_SX * (m_SY * _z + _y) + _x;
		}


		uint32_t m_SX, m_SY, m_SZ;
		uint32_t m_Padding;
	};
}
