#include "pch.h"
#include "NeuralNetwork.h"
#include "Util.h"
#include "BMP.h"
#include <chrono>
#include <fstream>
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


namespace ToyDNN
{

	void Dump( const Tensor& t )
	{
		for( Scalar f : t )
		{
			Log( "%f ", f );
		}
		Log( "\n" );
	}

	uint32_t GetMostProbableClassIndex( const Tensor& _tensor )
	{
		Scalar best = -FLT_MAX;
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

	void NeuralNetwork::Train( const std::vector<Tensor>& _trainingSet,
								const std::vector<Tensor>& _trainingSetExpectedOutput,
								const std::vector<Tensor>& _validationSet,
								const std::vector<Tensor>& _validationSetExpectedOutput,
								uint32_t _numEpochs, uint32_t _batchSize, uint32_t _validationInterval,
								Scalar _learningRate, Scalar _errorTarget )
	{
		assert( _trainingSet.size() == _trainingSetExpectedOutput.size() );
		assert( _validationSet.size() == _validationSetExpectedOutput.size() );
		assert( _batchSize <= _trainingSet.size() );

		m_IsTraining = true;
		m_StopTraining = false;

		uint32_t numTrainingSamples = (uint32_t)_trainingSet.size();
		uint32_t numValidationSamples = (uint32_t)_trainingSet.size();

		Tensor out;

		for( uint32_t epoch = 0; (epoch < _numEpochs) && !m_StopTraining ; ++epoch )
		{
			for( uint32_t batch = 0 ; (batch < numTrainingSamples / _batchSize) && !m_StopTraining ; ++batch )
			{
				ClearWeightDeltas();

				uint32_t batchSize = std::min( _batchSize, numTrainingSamples - batch * _batchSize );

				Scalar trainingError = Scalar(0.0);

				auto start = std::chrono::steady_clock::now();

				for( uint32_t batchSample = 0 ; batchSample < batchSize ; ++batchSample )
				{
					uint32_t sampleIndex = batch * _batchSize + batchSample;

					Evaluate( _trainingSet[sampleIndex], out );
					trainingError += ComputeError( out, _trainingSetExpectedOutput[sampleIndex] );
					BackPropagation( _trainingSet[sampleIndex], _trainingSetExpectedOutput[sampleIndex] );
				}

				auto end = std::chrono::steady_clock::now();
				float elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;

				trainingError /= batchSize;
				//Log( "epoch %d batch %d (%d samples) took %.1fs, error: %f\n", epoch, batch, batchSize, elapsedTime, error );

				Scalar fEpoch = Scalar( m_History.NumEpochCompleted ) + Scalar(batch + 1) / Scalar(numTrainingSamples / _batchSize);

				m_History.TrainingSetErrorXAxis.push_back( fEpoch );
				m_History.TrainingSetError.push_back( trainingError );

				ApplyWeightDeltas( _learningRate/*TODO divide by batch size ?*/ );

				//Every N batches or last batch of epoch
				bool bEvaluateValidationSet = (batch % _validationInterval == 0) || (batch == (numTrainingSamples / _batchSize) - 1);

				if( bEvaluateValidationSet )
				{
					Scalar validationSetError = ComputeError( _validationSet, _validationSetExpectedOutput );

					m_History.ValidationSetErrorXAxis.push_back( fEpoch );
					m_History.ValidationSetError.push_back( validationSetError );

					//Log( "Validation set error: %f\n", validationSetError );

					if( validationSetError <= _errorTarget )
					{
						m_IsTraining = false;
						return;
					}
				}
			}

			m_History.NumEpochCompleted++;
		}

		m_IsTraining = false;
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
				tensorIn = &tmpTensor[1 - curTensor];
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

			tensorOut->resize( m_Layers[layer]->GetOutputShape().Size() );

			m_Layers[layer]->Forward( *tensorIn, *tensorOut );
			m_Layers[layer]->CacheOutput( *tensorOut ); //For back propagation

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
			Scalar e = _out[i] - _expectedOutput[i];
			_error[i] = e * e * Scalar(0.5);
		}
	}

	Scalar NeuralNetwork::ComputeError( const Tensor& _out, const Tensor& _expectedOutput )
	{
		Scalar error = 0.0;

		#pragma omp parallel for reduction(+:error)
		for( int i = 0 ; i < (int)_out.size() ; ++i )
		{
			Scalar e = _out[i] - _expectedOutput[i];
			error += e * e * Scalar(0.5);
		}

		return error;//TODO average ?
	}

	Scalar NeuralNetwork::ComputeError( const std::vector<Tensor>& _dataSet, const std::vector<Tensor>& _dataSetExpectedOutput )
	{
	//#define classification_accurary

		Scalar error = 0.0;
		uint32_t validClassificationCount = 0;

		#pragma omp parallel for reduction(+:error)
		for( int i = 0 ; i < (int)_dataSet.size() ; ++i )
		{
			Tensor out;
			Evaluate( _dataSet[i], out );

			if( m_EnableClassificationAccuracyLog )
			{
				if( GetMostProbableClassIndex( _dataSetExpectedOutput[i] ) == GetMostProbableClassIndex( out ) )
					++validClassificationCount;
			}

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

		if( m_EnableClassificationAccuracyLog )
		{
			Log( "accuracy: %.1f%%\n", 100.0f * (float)validClassificationCount / (float)_dataSet.size() );
		}

		return error / (float)_dataSet.size();
	}


	void NeuralNetwork::BackPropagation( const Tensor& _input, const Tensor& _expectedOutput )
	{
		Tensor outputGradients[2];
		uint32_t curTensor = 0;

		//Compute Cost gradients, in other words dE/dA for last layer

		const uint32_t numOutputs = (uint32_t)_expectedOutput.size();
		outputGradients[0].resize( numOutputs );
		const Tensor& output = m_Layers[m_Layers.size() - 1]->GetOutput();

		assert( numOutputs == output.size() );

		for( uint32_t i = 0 ; i < numOutputs ; ++i )
		{
			//TODO make choice of Cost function a paramater to the network
			Scalar dE_dA = output[i] - _expectedOutput[i];
			outputGradients[0][i] = dE_dA;
		}

		for( int layer = (int)m_Layers.size() - 1 ; layer >= 0 ; --layer )
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

			outputGradients[1 - curTensor].resize( m_Layers[layer]->GetInputShape().Size() );

			m_Layers[layer]->BackPropagation( *tensorIn, outputGradients[curTensor], outputGradients[1 - curTensor] );

			curTensor = 1 - curTensor; //Ping pong
		}
	}

	void NeuralNetwork::ApplyWeightDeltas( Scalar _learningRate )
	{
		for( auto& layer : m_Layers )
			layer->ApplyWeightDeltas( _learningRate );
	}

	void NeuralNetwork::ClearHistory()
	{
		m_History.TrainingSetErrorXAxis.clear();
		m_History.TrainingSetError.clear();
		m_History.ValidationSetErrorXAxis.clear();
		m_History.ValidationSetError.clear();

		m_History.NumEpochCompleted = 0;
	}

	bool NeuralNetwork::Load( const char* _filename )
	{
		std::ifstream fileStream( _filename, std::ios::in | std::ios::binary );

		if( !fileStream.good() )
			return false;

		m_Layers.clear();

		try
		{
			ClearHistory();

			while( true )
			{
				LayerType layerType;
				Read( fileStream, layerType );

				if( fileStream.eof() )
					break;

				Layer* pLayer = CreateLayer( (LayerType)layerType );
				pLayer->Load( fileStream );

				m_Layers.push_back( std::unique_ptr<Layer>( pLayer ) );
			}
		
		}
		catch( ... )
		{
			Log( "Failed to load %s !\n", _filename );
			return false;
		}

		return true;
	}

	bool NeuralNetwork::Save( const char* _filename, bool _saveTrainingHistory ) const
	{
		std::ofstream fileStream( _filename, std::ios::out | std::ios::binary );

		if( !fileStream.good() )
			return false;

		try
		{
			//TODO if( _saveTrainingHistory )

			for( const auto& layer : m_Layers )
			{
				Write( fileStream, layer->GetType() );
				layer->Save( fileStream );
			}
		}
		catch( ... )
		{
			Log( "Failed to save %s !\n", _filename );
			return false;
		}

		return true;
	}

	void NeuralNetwork::GradientCheck( const std::vector<Tensor>& _dataSet, const std::vector<Tensor>& _dataSetExpectedOutput, uint32_t _numRandomParametersToCheck )
	{
		assert( _dataSet.size() == _dataSetExpectedOutput.size() );

		const Scalar epsilon = Scalar(1e-8);
		const Scalar gradientTolerance = Scalar(0.1);

		//Evaluate gradients through with back propagation

		ClearWeightDeltas();

		Tensor out;

		for( uint32_t i=0 ; i < _dataSet.size() ; ++i )
		{
			Evaluate( _dataSet[i], out );
			BackPropagation( _dataSet[i], _dataSetExpectedOutput[i] );
		}

		//Now compute "ground thruth" gradients with finite difference and compare them to back propagation gradients

		for( uint32_t i = 0 ; i < _numRandomParametersToCheck ; ++i )
		{
			//Pick a random parameter on a random layer
			uint32_t randomLayer;
			Scalar* pParameter;
			Scalar backPropGradient;

			do
			{
				randomLayer = rand() % m_Layers.size();
			} while( !m_Layers[randomLayer]->GetRandomParameterAndAssociatedGradient( &pParameter, backPropGradient ) );

			Scalar originalParamValue = *pParameter;
		
			//Compute Loss( param + epsilon )
			*pParameter = originalParamValue + epsilon;
			Scalar error1 = ComputeError( _dataSet, _dataSetExpectedOutput );

			//Compute Loss( param - epsilon )
			*pParameter = originalParamValue - epsilon;
			Scalar error2 = ComputeError( _dataSet, _dataSetExpectedOutput );

			//Compute gradient
			Scalar groundThruthGradient = (error1 - error2) / (2.0f * epsilon);

			Scalar gradientError = std::abs( (backPropGradient - groundThruthGradient) / groundThruthGradient );

			if( gradientError > gradientTolerance )
			{
			//	assert( false );
				Log( "Bad gradient (%f != %f, error=%.3f%%) found in layer %d. Probably a back propagation bug !\n", 
					 backPropGradient, groundThruthGradient, 100.0f * gradientError, randomLayer );
			}

			//restore the parameter
			*pParameter = originalParamValue;
		}
	}

}