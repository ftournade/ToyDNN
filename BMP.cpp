#include "BMP.h"

#include <Windows.h>
#include <vector>

bool WriteBMP(
	const char* fname,
	const uint8_t* buf,
	uint32_t width,
	uint32_t height
)
{
	uint32_t xb = (width * 3 + 3) & ~3;
	DWORD wb;
	BITMAPFILEHEADER bfh;
	BITMAPINFOHEADER bih;
	HANDLE hbmp;


	hbmp = CreateFileA( fname, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0 );
	if( hbmp == INVALID_HANDLE_VALUE ) 
		return false;

	memset( &bfh, 0, sizeof bfh );
	bfh.bfType = 'MB';
	bfh.bfSize = sizeof( BITMAPFILEHEADER ) + sizeof( BITMAPINFOHEADER ) + xb * height;
	bfh.bfOffBits = sizeof( BITMAPFILEHEADER ) + sizeof( BITMAPINFOHEADER );
	WriteFile( hbmp, &bfh, sizeof bfh, &wb, 0 );

	memset( &bih, 0, sizeof bih );
	bih.biSize = sizeof bih;
	bih.biWidth = width;
	bih.biHeight = height;
	bih.biPlanes = 1;
	bih.biBitCount = 24;
	bih.biCompression = BI_RGB;
	WriteFile( hbmp, &bih, sizeof bih, &wb, 0 );

	/* Flip top and down, swap byte order RGB to BGR */
#if 0
	uint8_t* s, * d, r, g, b;
	s = buf; d = buf + xb * (height - 1);
	while( s <= d )
	{
		for( uint32_t i = 0; i < width * 3; i += 3 )
		{
			r = s[i + 0]; g = s[i + 1]; b = s[i + 2];
			s[i + 0] = d[i + 2]; s[i + 1] = d[i + 1]; s[i + 2] = d[i + 0];
			d[i + 0] = b; d[i + 1] = g; d[i + 2] = r;
		}
		d -= xb; s += xb;
	}
	WriteFile( hbmp, buf, xb * height, &wb, 0 );
#else

	std::vector<uint8_t> buf2( xb * height );
	
	uint32_t s = 0;

	for( uint32_t y = 0 ; y < height ; ++y )
	{
		uint32_t d = (height - 1 - y) * xb;

		for( uint32_t x = 0 ; x < width ; ++x )
		{
			buf2[d++] = buf[s++];
			buf2[d++] = buf[s++];
			buf2[d++] = buf[s++];
		}
	}

	WriteFile( hbmp, &buf2[0], xb * height, &wb, 0 );
#endif

	CloseHandle( hbmp );

	return true;
}
