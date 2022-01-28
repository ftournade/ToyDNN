#include "pch.h"
#include "Win32BackBuffer.h"


Win32BackBuffer::Win32BackBuffer() :
	m_hBitmap(NULL),
	m_pPixels(NULL),
	m_Width(0),
	m_Height(0)
{
}

Win32BackBuffer::~Win32BackBuffer()
{
	Release();
}

bool Win32BackBuffer::Init( HWND _hWnd, int _width, int _height )
{
	Release();

	HDC hdc = GetDC(_hWnd);
	BITMAPINFO bi;
	bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bi.bmiHeader.biWidth = _width;
	bi.bmiHeader.biHeight = -_height; //negative to tell windows that (0,0) is top left
	bi.bmiHeader.biPlanes = 1; 
	bi.bmiHeader.biBitCount = 24; 
	bi.bmiHeader.biCompression = BI_RGB;
	bi.bmiHeader.biSizeImage = 0; 
	bi.bmiHeader.biXPelsPerMeter = 0; 
	bi.bmiHeader.biYPelsPerMeter = 0; 
	bi.bmiHeader.biClrUsed = 0; 
	bi.bmiHeader.biClrImportant = 0;

	m_hBitmap = CreateDIBSection( hdc, &bi, DIB_RGB_COLORS, &m_pPixels, NULL, NULL );

	if( !m_hBitmap || !m_pPixels )
		return false;

	m_hDC = CreateCompatibleDC( hdc );

	if( !m_hDC )
		return false;

	SelectObject( m_hDC, m_hBitmap );

	m_Width = _width;
	m_Height = _height;

	m_Pitch = ((_width * 24 + 31) & ~31) >> 3;

	//u32 padCount = NeededPaddingToAlign( _width * 3, 4 );

	return true;
}

void Win32BackBuffer::Release()
{
	DeleteObject( m_hBitmap );
	DeleteObject( m_hDC );
	m_hBitmap = NULL;
	m_hDC = NULL;
	m_pPixels = NULL;
	m_Width = 0;
	m_Height = 0;
}



bool Win32BackBuffer::Blit( HDC hdc )
{
	if( !BitBlt( hdc, 0, 0, m_Width, m_Height, m_hDC, 0, 0, SRCCOPY ) )
		return false;

	GdiFlush();

	return true;
}
