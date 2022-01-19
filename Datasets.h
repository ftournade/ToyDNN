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
