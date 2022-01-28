#include "pch.h"
#include "JPEG.h"
#include "ThirdParty/jpeg/tjpgd.h"
#include <windows.h>
#include <memory>

#define MODE	0	/* Test mode: 0:Show decmpression status, 1:and output in BMP */
#define SCALE	0	/* Output scaling 0:1/1, 1:1/2, 2:1/4 or 3:1/8 */


/*---------------------------------*/
/* User defined session identifier */
/*---------------------------------*/

typedef struct
{
	HANDLE hin;			/* Handle to the input stream */
	uint8_t* frmbuf;	/* Pointer to the frame buffer */
	uint32_t wbyte;		/* Number of bytes a line in the frame buffer */
} IODEV;

size_t jpeg_input_func(	/* Returns number of bytes read (zero on error) */
				   JDEC* jd,		/* Decompression object */
				   uint8_t* buff,	/* Pointer to the read buffer (null to remove data) */
				   size_t ndata	/* Number of bytes to read/skip */
)
{
	DWORD rb;
	IODEV* dev = (IODEV*)jd->device;


	if( buff )
	{	/* Read bytes from input stream */
		ReadFile( dev->hin, buff, ndata, &rb, 0 );
		return (size_t)rb;
	}
	else
	{	/* Remove bytes from input stream */
		rb = SetFilePointer( dev->hin, ndata, 0, FILE_CURRENT );
		return rb == 0xFFFFFFFF ? 0 : ndata;
	}
}


int jpeg_output_func(	/* 1:Ok, 0:Aborted */
				 JDEC* jd,		/* Decompression object */
				 void* bitmap,	/* Bitmap data to be output */
				 JRECT* rect		/* Rectangular region to output */
)
{
	uint32_t nx, ny, xc, wd;
	uint8_t* src, * dst;
	IODEV* dev = (IODEV*)jd->device;


	nx = rect->right - rect->left + 1;
	ny = rect->bottom - rect->top + 1;	/* Number of lines of the rectangular */
	src = (uint8_t*)bitmap;				/* RGB bitmap to be output */

	wd = dev->wbyte;							/* Number of bytes a line of the frame buffer */
	dst = dev->frmbuf + rect->top * wd + rect->left * 3;	/* Left-top of the destination rectangular in the frame buffer */

	do
	{	/* Copy the rectangular to the frame buffer */
		xc = nx;
		do
		{
			if( JD_FORMAT == 2 )
			{	/* Grayscale output */
				*dst++ = *src;
				*dst++ = *src;
				*dst++ = *src;
				src++;
			}
			else if( JD_FORMAT == 1 )
			{	/* RGB565 output */
				*dst++ = (src[1] & 0xF8) | src[1] >> 5;
				*dst++ = src[1] << 5 | (src[0] & 0xE0) >> 3 | (src[1] >> 1 & 3);
				*dst++ = src[0] << 3 | (src[0] & 0x1F) >> 2;
				src += 2;
			}
			else
			{				/* RGB888 output */
				*dst++ = *src++;
				*dst++ = *src++;
				*dst++ = *src++;
			}
		} while( --xc );
		dst += wd - nx * 3;
	} while( --ny );

	return 1;	/* Continue to decompress */
}

bool LoadJpeg( const char* fname, std::vector< uint8_t >& _pixels, uint32_t& _width, uint32_t& _height )
{
	const size_t sz_work = 32768;	/* Size of working buffer for TJpgDec module */
	JDEC jd;		/* TJpgDec decompression object */
	IODEV iodev;	/* Identifier of the decompression session (depends on application) */
	JRESULT rc;
	uint32_t xb, xs, ys;
	
	/* Open JPEG file */
	iodev.hin = CreateFileA( fname, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0 ); //TODO RAII
	if( iodev.hin == INVALID_HANDLE_VALUE )
		return false;

	auto jdwork = std::make_unique< uint8_t[] >( sz_work );

	/* Prepare to decompress the JPEG image */
	rc = jd_prepare( &jd, jpeg_input_func, jdwork.get(), sz_work, &iodev );

	if( rc != JDR_OK )
		return false;

	/* Initialize frame buffer */
	xs = jd.width >> SCALE;		/* Image size to output */
	ys = jd.height >> SCALE;
	//xb = (xs * 3 + 3) & ~3;		/* Byte width of the frame buffer */
	xb = xs * 3;
	iodev.wbyte = xb;

	_pixels.resize( xb * ys );
	iodev.frmbuf = &_pixels[0];

	/* Start JPEG decompression */
	rc = jd_decomp( &jd, jpeg_output_func, SCALE );

	CloseHandle( iodev.hin );	/* Close JPEG file */

	_width = jd.width;
	_height = jd.height;

	return true;
}