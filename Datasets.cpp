#include "Datasets.h"
#include "JPEG.h"
#include "BMP.h" //test
#include "Util.h"
#include <assert.h>



bool LoadJpeg( const char* _filename, bool _grayscale, Tensor& _pixels )
{
	std::vector< uint8_t > rgbPixels; //RGB8

	if( !LoadJpeg( _filename, rgbPixels ) )
		return false;

	uint32_t numPixels = rgbPixels.size() / 3;

	if( _grayscale )
	{
		_pixels.resize( numPixels );

		for( uint32_t i = 0 ; i < numPixels ; ++i )
		{
			float p = (float)rgbPixels[i * 3] + (float)rgbPixels[i * 3 + 1] + (float)rgbPixels[i * 3 + 2];
			p /= 3.0f;
			_pixels[i] = p / 255.0f;
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
			uint8_t p = _pixels[i] * 255.0f; //TODO saturate
			data[i * 3] = p;
			data[i * 3 + 1] = p;
			data[i * 3 + 2] = p;
		}
	}

	return WriteBMP( _filename, &data[0], _width, _height );
}

const uint32_t CelebADatasetSize = 202599;

bool LoadCelebADataset( const char* _filepath,
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

		if( !LoadJpeg( filename, true, _data[i] ) )
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
						float _trainingSetRatio, //e.g. 0.5 loads 50% of the database
						float _validationSetRatio,
						std::vector< Tensor >& _trainingSetData,
						std::vector< Tensor >& _validationSetData,
						std::vector< CelebAMetaData >& _trainingSetMetaData,
						std::vector< CelebAMetaData >& _validationSetMetaData )
{
	
	assert( _trainingSetRatio > 0.0f && _trainingSetRatio <= 1.0f );
	assert( _validationSetRatio > 0.0f && _validationSetRatio <= 1.0f );
	assert( _trainingSetRatio + _validationSetRatio <= 1.0f );

	uint32_t trainingSetSize = (uint32_t)((float)CelebADatasetSize * _trainingSetRatio);
	uint32_t validationSetSize = (uint32_t)((float)CelebADatasetSize * _validationSetRatio);

	if( !LoadCelebADataset( _filepath, 1, trainingSetSize, _trainingSetData, _trainingSetMetaData ) )
		return false;

	if( !LoadCelebADataset( _filepath, trainingSetSize + 1, validationSetSize, _validationSetData, _validationSetMetaData ) )
		return false;

	return true;
}