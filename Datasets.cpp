#include "Datasets.h"
#include "JPEG.h"
#include "BMP.h" //test
#include "Util.h"
#include <assert.h>



bool LoadJpeg( const char* _filename, bool _halfRes, bool _grayscale, Tensor& _pixels )
{
	std::vector< uint8_t > rgbPixels; //RGB8
	uint32_t width, height;

	if( !LoadJpeg( _filename, rgbPixels, width, height ) )
		return false;


	if( _grayscale )
	{
		auto Grayscale = [&]( uint32_t x, uint32_t y ) {
			uint32_t idx = (y * width + x) * 3;
			return ((float)rgbPixels[idx] + (float)rgbPixels[idx + 1] + (float)rgbPixels[idx + 2]) / (3.0f * 255.0f); };

		if( _halfRes )
		{
			assert( (width % 2 == 0) && (height % 2 == 0) ); //TODO handle odd resolutions
			_pixels.resize( width * height / 4 );

			for( uint32_t y = 0 ; y < height / 2 ; ++y )
			{
				for( uint32_t x = 0 ; x < width / 2 ; ++x )
				{
					//Average 4 pixels
					float p = Grayscale( x*2, y*2 ) + Grayscale( x*2+1, y*2 ) + Grayscale( x*2, y*2+1 ) + Grayscale( x*2+1, y*2+1 );
					p *= 0.25f;

					_pixels[y * width / 2 + x] = p;
				}
			}
		}
		else
		{
			_pixels.resize( rgbPixels.size() / 3 );

			for( uint32_t y = 0 ; y < height ; ++y )
			{
				for( uint32_t x = 0 ; x < width ; ++x )
				{
					_pixels[y*width + x] = Grayscale( x, y );
				}
			}
		}
	}
	else
	{
		return false; //TODO
	}

	return true;
}

bool WriteBMP( const char* _filename, bool _grayscale, const Tensor& _pixels, int _width, int _height )
{
	std::vector<uint8_t> data;

	if( _grayscale )
	{
		data.resize( _pixels.size() * 3 );
	
		for( uint32_t i = 0 ; i < _pixels.size() ; ++i )
		{
			uint8_t p =(uint8_t)( _pixels[i] * 255.0f ); //TODO saturate
			data[i * 3] = p;
			data[i * 3 + 1] = p;
			data[i * 3 + 2] = p;
		}
	}

	return WriteBMP( _filename, &data[0], _width, _height );
}

const uint32_t CelebADatasetSize = 202599;

bool LoadCelebADataset( const char* _filepath,
						bool _halfRes,
						uint32_t _firstSample, 
						uint32_t _numSamples,
						std::vector< Tensor >& _data,
						std::vector< CelebAMetaData >& _metaData )

{
	assert( _firstSample >= 1 );
	assert( _firstSample + _numSamples - 1 <= CelebADatasetSize );

	_data.resize( _numSamples );

	for( uint32_t i = 0 ; i < _numSamples ; ++i )
	{
		char filename[1024];
		sprintf_s( filename, "%s\\%06d.jpg", _filepath, _firstSample + i );

		if( !LoadJpeg( filename, _halfRes, true, _data[i] ) )
			return false;

		if( (i > 0) && (i % 50 == 0) )
		{
			Log( "JPEG %d/%d loaded.\n", i, _numSamples );
		}
	}

	Log( "JPEG %d/%d loaded.\n", _numSamples, _numSamples );

	return true;
}

bool LoadCelebADataset( const char* _filepath,
						bool _halfRes,
						float _trainingSetRatio, //in percent
						float _validationSetRatio, //in percent
						std::vector< Tensor >& _trainingSetData,
						std::vector< Tensor >& _validationSetData,
						std::vector< CelebAMetaData >& _trainingSetMetaData,
						std::vector< CelebAMetaData >& _validationSetMetaData )
{
	
	_trainingSetRatio /= 100.0f;
	_validationSetRatio /= 100.0f;

	assert( _trainingSetRatio > 0.0f && _trainingSetRatio <= 1.0f );
	assert( _validationSetRatio > 0.0f && _validationSetRatio <= 1.0f );
	assert( _trainingSetRatio + _validationSetRatio <= 1.0f );

	uint32_t trainingSetSize = (uint32_t)((float)CelebADatasetSize * _trainingSetRatio);
	uint32_t validationSetSize = (uint32_t)((float)CelebADatasetSize * _validationSetRatio);

	if( !LoadCelebADataset( _filepath, _halfRes, 1, trainingSetSize, _trainingSetData, _trainingSetMetaData ) )
		return false;

	if( !LoadCelebADataset( _filepath, _halfRes, trainingSetSize + 1, validationSetSize, _validationSetData, _validationSetMetaData ) )
		return false;

	return true;
}