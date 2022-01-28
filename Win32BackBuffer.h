#pragma once

class Win32BackBuffer
{
public:
	Win32BackBuffer();
	~Win32BackBuffer();

	bool Init( HWND _hWnd, int _width, int _height );
	void Release();

	bool Blit( HDC hdc );
		
	inline HDC GetBackBufferDC() { return m_hDC; }

	inline int GetWidth() const { return m_Width; }
	inline int GetHeight() const { return m_Height; }

//	inline void SetPixel( int x, int y, const Vec3& _col );
//	inline void SetPixel( int x, int y, const Color& _col );
	
	inline const byte* GetPixels() const { return (const byte*)m_pPixels; }
		

protected:
	int		m_Width, m_Height, m_Pitch;
	HBITMAP m_hBitmap;
	HDC		m_hDC;
	void*	m_pPixels;

};
/*
//TODO move in Core
#ifdef _M_IX86
__forceinline s32 float2int( float f )
{
	s32 integer;
	_asm
	{
		fld f
		fistp integer	
	}
	return integer;
}
#else
#define float2int( x ) ((s32)(x))
#endif

__forceinline void Win32BackBuffer::SetPixel( int x, int y, const Color& _col )
{
	DBG_CHECK( (x < m_Width) && (y < m_Height) );
		
	//TODO use ToWin32COLORREF
	//y = m_Height - 1 - y;
	u8* pPixel = (u8*)m_pPixels + y * m_Pitch + x * 3;
	*pPixel++ = (u8)float2int(_col.b * 255.0f);
	*pPixel++ = (u8)float2int(_col.g * 255.0f);
	*pPixel   = (u8)float2int(_col.r * 255.0f);
}

__forceinline void Win32BackBuffer::SetPixel( int x, int y, const Vec3& _col )
{
	DBG_CHECK( (x < m_Width) && (y < m_Height) );

	//y = m_Height - 1 - y;
	u8* pPixel = (u8*)m_pPixels + y * m_Pitch + x * 3;
	*pPixel++ = (u8)(_col.z * 255.0f);
	*pPixel++ = (u8)(_col.y * 255.0f);
	*pPixel   = (u8)(_col.x * 255.0f);
}
*/