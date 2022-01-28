#pragma once

#include "Tensor.h"


struct CelebAMetaData
{

};

bool LoadCelebADataset( const char* _filepath,
						bool _halfRes, 
						float _trainingSetRatio, //e.g. 0.5 loads 50% of the database
						float _validationSetRatio,
						std::vector< Tensor >& _trainingSetData,
						std::vector< Tensor >& _validationSetData,
						std::vector< CelebAMetaData >& _trainingSetMetaData,
						std::vector< CelebAMetaData >& _validationSetMetaData );


bool LoadMnistDataset(	const char* _filepath,
						float_t _scale_min,
						float_t _scale_max,
						int _x_padding,
						int _y_padding,
						std::vector< Tensor >& _trainingSetData,
						std::vector< Tensor >& _validationSetData,
						std::vector< Tensor >& _trainingSetMetaData,
						std::vector< Tensor >& _validationSetMetaData );

bool LoadCifar10Dataset( const char* _filepath,
						 std::vector< Tensor >& _trainingSetData,
						 std::vector< Tensor >& _validationSetData,
						 std::vector< Tensor >& _trainingSetMetaData,
						 std::vector< Tensor >& _validationSetMetaData );
