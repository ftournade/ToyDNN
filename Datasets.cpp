#include "Datasets.h"
#include "JPEG.h"
#include "BMP.h" //test
#include "Util.h"
#include <assert.h>
#include <fstream>

namespace ToyDNN
{

#pragma region Util

	template <typename T>
	T* reverse_endian( T* p )
	{
		std::reverse( reinterpret_cast<char*>(p),
					  reinterpret_cast<char*>(p) + sizeof( T ) );
		return p;
	}

	inline bool is_little_endian()
	{
		int x = 1;
		return *reinterpret_cast<char*>(&x) != 0;
	}

	bool LoadJpeg( const char* _filename, bool _halfRes, bool _grayscale, Tensor& _pixels )
	{
		std::vector< uint8_t > rgbPixels; //RGB8
		uint32_t width, height;

		if( !::LoadJpeg( _filename, rgbPixels, width, height ) )
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
						float p = Grayscale( x * 2, y * 2 ) + Grayscale( x * 2 + 1, y * 2 ) + Grayscale( x * 2, y * 2 + 1 ) + Grayscale( x * 2 + 1, y * 2 + 1 );
						p *= 0.25f;

						_pixels[y * width / 2 + x] = p;
					}
				}
			}
			else
			{
				_pixels.resize( width * height );

				for( uint32_t y = 0 ; y < height ; ++y )
				{
					for( uint32_t x = 0 ; x < width ; ++x )
					{
						_pixels[y * width + x] = Grayscale( x, y );
					}
				}
			}
		}
		else
		{
			auto AddColor = [&]( uint32_t x, uint32_t y, float& r, float& g, float& b ) {
				uint32_t idx = (y * width + x) * 3;
				r += (float)rgbPixels[idx] / 255.0f;
				g += (float)rgbPixels[idx + 1] / 255.0f;
				b += (float)rgbPixels[idx + 2] / 255.0f;
			};

			if( _halfRes )
			{
				assert( (width % 2 == 0) && (height % 2 == 0) ); //TODO handle odd resolutions
				uint32_t channelSize = width * height / 4;
				_pixels.resize( channelSize * 3 );

				for( uint32_t y = 0 ; y < height / 2 ; ++y )
				{
					for( uint32_t x = 0 ; x < width / 2 ; ++x )
					{
						//Average 4 pixels
						float r = 0.0f, g = 0.0f, b = 0.0f;

						AddColor( x * 2, y * 2, r, g, b );

						r *= 0.25f;
						g *= 0.25f;
						b *= 0.25f;

						//Deinterlace RGB
						_pixels[y * width / 2 + x] = r;
						_pixels[channelSize + y * width / 2 + x] = g;
						_pixels[channelSize * 2 + y * width / 2 + x] = b;
					}
				}
			}
			else
			{
				uint32_t channelSize = width * height;
				_pixels.resize( channelSize * 3 );

				for( uint32_t y = 0 ; y < height ; ++y )
				{
					for( uint32_t x = 0 ; x < width ; ++x )
					{
						float r = 0.0f, g = 0.0f, b = 0.0f;

						AddColor( x, y, r, g, b );

						//Deinterlace RGB
						_pixels[y * width / 2 + x] = r;
						_pixels[channelSize + y * width / 2 + x] = g;
						_pixels[channelSize * 2 + y * width / 2 + x] = b;
					}
				}
			}
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
				uint8_t p = (uint8_t)(_pixels[i] * 255.0f); //TODO saturate
				data[i * 3] = p;
				data[i * 3 + 1] = p;
				data[i * 3 + 2] = p;
			}
		}
		else
		{
			uint32_t numPixels = _pixels.size() / 3;
			data.resize( numPixels * 3 );

			for( uint32_t i = 0 ; i < numPixels; ++i )
			{
				uint8_t r = (uint8_t)(_pixels[i] * 255.0f); //TODO saturate
				uint8_t g = (uint8_t)(_pixels[i + numPixels] * 255.0f); //TODO saturate
				uint8_t b = (uint8_t)(_pixels[i + numPixels * 2] * 255.0f); //TODO saturate
				data[i * 3] = b;
				data[i * 3 + 1] = g;
				data[i * 3 + 2] = r;
			}
		}

		return ::WriteBMP( _filename, &data[0], _width, _height );
	}
#pragma endregion

#pragma region CelebA

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

			if( !LoadJpeg( filename, _halfRes, false, _data[i] ) )
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
#pragma endregion

#pragma region MNIST

	//Code borrowed from tiny_dnn

	struct mnist_header
	{
		uint32_t magic_number;
		uint32_t num_items;
		uint32_t num_rows;
		uint32_t num_cols;
	};

	bool ParseMnistHeader( std::ifstream& ifs, mnist_header& header )
	{
		ifs.read( reinterpret_cast<char*>(&header.magic_number), 4 );
		ifs.read( reinterpret_cast<char*>(&header.num_items), 4 );
		ifs.read( reinterpret_cast<char*>(&header.num_rows), 4 );
		ifs.read( reinterpret_cast<char*>(&header.num_cols), 4 );

		if( is_little_endian() )
		{
			reverse_endian( &header.magic_number );
			reverse_endian( &header.num_items );
			reverse_endian( &header.num_rows );
			reverse_endian( &header.num_cols );
		}

		if( header.magic_number != 0x00000803 || header.num_items <= 0 )
			return false;

		if( ifs.fail() || ifs.bad() )
			return false;

		return true;
	}

	bool LoadMnistImage( std::ifstream& ifs,
						 const mnist_header& header,
						 float_t scale_min,
						 float_t scale_max,
						 int x_padding,
						 int y_padding,
						 Tensor& dst )
	{
		const int width = header.num_cols + 2 * x_padding;
		const int height = header.num_rows + 2 * y_padding;

		std::vector<uint8_t> image_vec( header.num_rows * header.num_cols );

		ifs.read( reinterpret_cast<char*>(&image_vec[0]),
				  header.num_rows * header.num_cols );

		dst.resize( width * height, scale_min );

		for( uint32_t y = 0; y < header.num_rows; y++ )
			for( uint32_t x = 0; x < header.num_cols; x++ )
				dst[width * (y + y_padding) + x + x_padding] =
				(image_vec[y * header.num_cols + x] / float_t( 255 )) *
				(scale_max - scale_min) +
				scale_min;

		return true;
	}

	bool LoadMnistLabels( const char* _filename, std::vector<Tensor>& _labels )
	{
		std::ifstream ifs( _filename, std::ios::in | std::ios::binary );

		if( ifs.bad() || ifs.fail() )
			return false;

		uint32_t magic_number, num_items;

		ifs.read( reinterpret_cast<char*>(&magic_number), 4 );
		ifs.read( reinterpret_cast<char*>(&num_items), 4 );

		if( is_little_endian() )
		{  // Mnist data is big-endian format
			reverse_endian( &magic_number );
			reverse_endian( &num_items );
		}

		if( magic_number != 0x00000801 || num_items <= 0 )
			return false;

		_labels.resize( num_items );
		for( uint32_t i = 0; i < num_items; i++ )
		{
			uint8_t label;
			ifs.read( reinterpret_cast<char*>(&label), 1 );

			_labels[i].resize( 10, 0 );
			_labels[i][label] = 1;
		}

		return true;
	}

	/**
	 * parse Mnist database format images with rescaling/resizing
	 * http://yann.lecun.com/exdb/mnist/
	 * - if original image size is WxH, output size is
	 *(W+2*x_padding)x(H+2*y_padding)
	 * - extra padding pixels are filled with scale_min
	 *
	 * @param image_file [in]  filename of database (i.e.train-images-idx3-ubyte)
	 * @param images     [out] parsed image data
	 * @param scale_min  [in]  min-value of output
	 * @param scale_max  [in]  max-value of output
	 * @param x_padding  [in]  adding border width (left,right)
	 * @param y_padding  [in]  adding border width (top,bottom)
	 *
	 * [example]
	 * scale_min=-1.0, scale_max=1.0, x_padding=1, y_padding=0
	 *
	 * [input]       [output]
	 *  64  64  64   -1.0 -0.5 -0.5 -0.5 -1.0
	 * 128 128 128   -1.0  0.0  0.0  0.0 -1.0
	 * 255 255 255   -1.0  1.0  1.0  1.0 -1.0
	 *
	 **/

	bool LoadMnistImages( const char* _filename,
						  std::vector<Tensor>& _images,
						  float_t scale_min,
						  float_t scale_max,
						  int x_padding,
						  int y_padding )
	{
		if( x_padding < 0 || y_padding < 0 )
			return false;

		if( scale_min >= scale_max )
			return false;

		std::ifstream ifs( _filename, std::ios::in | std::ios::binary );

		if( ifs.bad() || ifs.fail() )
			return false;

		mnist_header header;
		if( !ParseMnistHeader( ifs, header ) )
			return false;

		_images.resize( header.num_items );
		for( uint32_t i = 0; i < header.num_items; i++ )
		{
			if( !LoadMnistImage( ifs, header, scale_min, scale_max, x_padding, y_padding, _images[i] ) )
				return false;
		}

		return true;
	}

	bool LoadMnistDataset( const char* _filepath,
						   float_t _scale_min,
						   float_t _scale_max,
						   int _x_padding,
						   int _y_padding,
						   std::vector< Tensor >& _trainingSetData,
						   std::vector< Tensor >& _validationSetData,
						   std::vector< Tensor >& _trainingSetMetaData,
						   std::vector< Tensor >& _validationSetMetaData )
	{
		char filename[1024];
		sprintf_s( filename, "%s\\train-images.idx3-ubyte", _filepath );

		if( !LoadMnistImages( filename, _trainingSetData, _scale_min, _scale_max, _x_padding, _y_padding ) )
			return false;

		sprintf_s( filename, "%s\\t10k-images.idx3-ubyte", _filepath );

		if( !LoadMnistImages( filename, _validationSetData, _scale_min, _scale_max, _x_padding, _y_padding ) )
			return false;

		sprintf_s( filename, "%s\\train-labels.idx1-ubyte", _filepath );

		if( !LoadMnistLabels( filename, _trainingSetMetaData ) )
			return false;

		sprintf_s( filename, "%s\\t10k-labels.idx1-ubyte", _filepath );

		if( !LoadMnistLabels( filename, _validationSetMetaData ) )
			return false;

		return true;
	}


#pragma endregion

#pragma region CIFAR10

	bool LoadCifar10Dataset( const char* _filename,
							 std::vector< Tensor >& _dataSet,
							 std::vector< Tensor >& _metaData )
	{
		static const uint32_t numPixels = 32 * 32;

		std::ifstream ifs( _filename, std::ios::in | std::ios::binary );

		if( ifs.fail() || ifs.bad() )
			return false;

		uint8_t label;
		uint8_t pixels[numPixels * 3];

		Tensor image( numPixels * 3 );
		Tensor labelTensor( 10 );

		while( ifs.read( (char*)&label, 1 ) )
		{

			if( !ifs.read( (char*)(&pixels[0]), numPixels * 3 ) )
				break;


			std::fill( labelTensor.begin(), labelTensor.end(), 0.0f );
			labelTensor[label] = 1.0f;

			for( uint32_t i = 0 ; i < numPixels * 3 ; ++i )
			{
				image[i] = (float)pixels[i] / 255.0f;
			}

			_dataSet.push_back( image );
			_metaData.push_back( labelTensor );
		}

		return true;
	}


	bool LoadCifar10Dataset( const char* _filepath,
							 std::vector< Tensor >& _trainingSetData,
							 std::vector< Tensor >& _validationSetData,
							 std::vector< Tensor >& _trainingSetMetaData,
							 std::vector< Tensor >& _validationSetMetaData )
	{
		char filename[1024];
		sprintf_s( filename, "%s\\test_batch.bin", _filepath );

		if( !LoadCifar10Dataset( filename, _validationSetData, _validationSetMetaData ) )
			return false;

		for( int i = 0 ; i < 5 ; ++i )
		{
			sprintf_s( filename, "%s\\data_batch_%d.bin", _filepath, i + 1 );

			if( !LoadCifar10Dataset( filename, _trainingSetData, _trainingSetMetaData ) )
				return false;
		}

		return true;
	}

#pragma endregion

}