#include "NeuralNetwork.h"
#include "Util.h"
#include "BMP.h"
#include <chrono>
/*
Back-propagation
	E: error
	Y: expected output
	A: neuron output (post activation)
	N: net neuron input (pre activation)
	W: connection weight
	B: neuron bias

	E = 1/2 * (A-Y)² = (A² -2AY + Y²)/2
	A = Sigmoid( N )

	----------------------
	dE/dW = dE/dA * dA/dN * dN/dW

	dE/dA = (2A - 2Y)/2 = (A - Y)
	dA/dN = Sigmoid( N ) * (1.0f - Sigmoid( N ))
	dN/dW = A'  (activation from layer l-1)
	----------------------
	dE/dB = dE/dA * dA/dN * dN/dB
	dN/dB = 1
	
*/
void Dump( const Tensor& t )
{
	for( float f : t )
	{
		Log( "%f ", f );
	}
	Log( "\n" );
}

uint32_t GetMostProbableClassIndex( const Tensor& _tensor )
{
	float best = -FLT_MAX;
	uint32_t bestIdx;

	for( int i = 0 ; i < _tensor.size() ; ++i )
	{
		if( _tensor[i] > best )
		{
			best = _tensor[i];
			bestIdx = i;
		}
	}

	return bestIdx;
}

void NeuralNetwork::AddLayer( Layer* _layer ) 
{
	m_Layers.push_back( std::unique_ptr<Layer>( _layer ) ); 
}

void NeuralNetwork::Compile( const TensorShape& _inputShape )
{
	assert( !m_Layers.empty() );

	m_Layers[0]->Setup( _inputShape );

	for( uint32_t i = 1 ; i < m_Layers.size() ; ++i )
	{
		m_Layers[i]->Setup( m_Layers[i - 1]->GetOutputShape() );
	}
}

float NeuralNetwork::Train( const std::vector<Tensor>& _trainingSet,
							const std::vector<Tensor>& _trainingSetExpectedOutput,
							const std::vector<Tensor>& _validationSet,
							const std::vector<Tensor>& _validationSetExpectedOutput,
							uint32_t _numEpochs, uint32_t _batchSize, uint32_t _validationInterval, 
							float _learningRate, float _errorTarget )
{
	assert( _trainingSet.size() == _trainingSetExpectedOutput.size() );
	assert( _validationSet.size() == _validationSetExpectedOutput.size() );
	uint32_t numTrainingSamples = _trainingSet.size();
	uint32_t numValidationSamples = _trainingSet.size();

	float error;
	Tensor out;

	for( uint32_t epoch = 0; epoch < _numEpochs ; ++epoch )
	{
		for( uint32_t batch = 0 ; batch < numTrainingSamples / _batchSize ; ++batch )
		{
			ClearWeightDeltas();

			uint32_t batchSize = std::min( _batchSize, numTrainingSamples - batch * _batchSize );

			error = 0.0f;
			
			auto start = std::chrono::steady_clock::now();
			
			for( uint32_t batchSample = 0 ; batchSample < batchSize ; ++batchSample )
			{
				uint32_t sampleIndex = batch * _batchSize + batchSample;

				Evaluate( _trainingSet[sampleIndex], out );
				error += ComputeError( out, _trainingSetExpectedOutput[sampleIndex] );
				BackPropagation( _trainingSet[sampleIndex], _trainingSetExpectedOutput[sampleIndex] );
			}

			auto end = std::chrono::steady_clock::now();
			float elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
			
			error /= batchSize;
			//Log( "epoch %d batch %d (%d samples) took %.1fs, error: %f\n", epoch, batch, batchSize, elapsedTime, error );

			ApplyWeightDeltas( _learningRate/*TODO divide by batch size ?*/);

			//Every N batches or last batch of epoch
			bool bEvaluateValidationSet = (batch % _validationInterval == 0) || (batch == (numTrainingSamples / _batchSize) - 1);

			if( bEvaluateValidationSet )
			{
				float validationSetError = ComputeError( _validationSet, _validationSetExpectedOutput );

				Log( "Validation set error: %f\n", validationSetError );

				if( validationSetError <= _errorTarget )
					return error;
			}
		}
	}

	return error;
}

void NeuralNetwork::Evaluate( const Tensor& _in, Tensor& _out ) const
{
	Tensor tmpTensor[2];
	uint32_t curTensor = 0;

	for( uint32_t layer = 0 ; layer < m_Layers.size() ; ++layer )
	{
		const Tensor* tensorIn;
		
		if( layer == 0 )
		{
			tensorIn = &_in;
		}
		else
		{
			tensorIn = &tmpTensor[1-curTensor];
		}

		Tensor* tensorOut;
		
		if( layer == m_Layers.size() - 1 )
		{
			tensorOut = &_out;
		}
		else
		{
			tensorOut = &tmpTensor[curTensor];
		}

		m_Layers[layer]->Forward( *tensorIn, *tensorOut );

		curTensor = 1 - curTensor;//ping pong
	}
}

void NeuralNetwork::ClearWeightDeltas()
{
	for( auto& layer : m_Layers )
		layer->ClearWeightDeltas();
}

void NeuralNetwork::ComputeError( const Tensor& _out, const Tensor& _expectedOutput, Tensor& _error )
{
	assert( _expectedOutput.size() == _out.size() );
	_error.resize( _out.size() );

	for( uint32_t i = 0 ; i < _out.size() ; ++i )
	{
		float e = _out[i] - _expectedOutput[i];
		_error[i] = e * e * 0.5f;
	}
}

float NeuralNetwork::ComputeError( const Tensor& _out, const Tensor& _expectedOutput )
{
	float error = 0.0f;

	#pragma omp parallel for reduction(+:error)
	for( int i = 0 ; i < (int)_out.size() ; ++i )
	{
		float e = _out[i] - _expectedOutput[i];
		error += e * e * 0.5f;
	}

	return error;//TODO average ?
}

float NeuralNetwork::ComputeError( const std::vector<Tensor>& _dataSet, const std::vector<Tensor>& _dataSetExpectedOutput )
{
#define classification_accurary

	float error = 0.0f;
	
#ifdef classification_accurary
	uint32_t validOutputCount = 0;
#endif

	#pragma omp parallel for reduction(+:error)
	for( int i = 0 ; i < (int)_dataSet.size() ; ++i )
	{
		Tensor out;
		Evaluate( _dataSet[i], out );

	#ifdef classification_accurary
		if( GetMostProbableClassIndex( _dataSetExpectedOutput[i] ) == GetMostProbableClassIndex( out ) )
			++validOutputCount;
	#endif

	#if 0 //Test
		if( i < 8 )
		{
			#define SX (178/2)
			#define SY (218/2)
			char buf[512];
			sprintf_s( buf, "d:\\tmp\\vs%d_in.bmp", i );
			WriteBMP( buf, false, _dataSet[i], SX, SY );
			sprintf_s( buf, "d:\\tmp\\vs%d_out.bmp", i );
			WriteBMP( buf, false, out, SX, SY );
			#undef SX
			#undef SY
		}
	#endif
		error += ComputeError( out, _dataSetExpectedOutput[i] );
	}

#ifdef classification_accurary
	Log( "accuracy: %.1f%%\n", 100.0f * (float)validOutputCount / (float)_dataSet.size() );
#endif

	return error / (float)_dataSet.size();
}


void NeuralNetwork::BackPropagation( const Tensor& _input, const Tensor& _expectedOutput )
{
	Tensor outputGradients[2];
	uint32_t curTensor = 0;

	//Compute Cost gradients, in other words dE/dA for last layer

	const uint32_t numOutputs = _expectedOutput.size();
	outputGradients[0].resize( numOutputs );
	const Tensor& output = m_Layers[m_Layers.size() - 1]->GetOutput();

	for( uint32_t i = 0 ; i < numOutputs ; ++i )
	{
		//TODO make choice of Cost function a paramater to the network
		float dE_dA = output[i] - _expectedOutput[i];
		outputGradients[0][i] = dE_dA;
	}

	for( int layer = (int)m_Layers.size() - 1 ; layer >= 0  ; --layer )
	{
		const Tensor* tensorIn;

		if( layer == 0 )
		{
			tensorIn = &_input;
		}
		else
		{
			tensorIn = &m_Layers[layer - 1]->GetOutput();
		}

		m_Layers[layer]->BackPropagation( *tensorIn, outputGradients[curTensor], outputGradients[1 - curTensor] );
		
		curTensor = 1 - curTensor; //Ping pong
	}
}

void NeuralNetwork::ApplyWeightDeltas( float _learningRate )
{
	for( auto& layer : m_Layers )
		layer->ApplyWeightDeltas( _learningRate );
}

