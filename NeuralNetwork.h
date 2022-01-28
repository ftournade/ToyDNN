#pragma once

#include "Layer.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/Convolution2DLayer.h"

#include <memory>

class NeuralNetwork
{
public:
	void AddLayer( std::unique_ptr<Layer> _layer );

	void Compile( const TensorShape& _inputShape );

	//Return error metric
	float Train( const std::vector<Tensor>& _trainingSet,
				 const std::vector<Tensor>& _trainingSetExpectedOutput,
				 const std::vector<Tensor>& _validationSet,
				 const std::vector<Tensor>& _validationSetExpectedOutput,
				 uint32_t _numEpochs, uint32_t _batchSize, uint32_t _validationInterval /*evaluate vaildationSet every N batch*/,
				 float _learningRate, float _errorTarget = 0.0001f );
	void Evaluate( const Tensor& _in, Tensor& _out ) const;

	static void ComputeError( const Tensor& _out, const Tensor& _expectedOutput, Tensor& _error );
	static float ComputeError( const Tensor& _out, const Tensor& _expectedOutput );
	float ComputeError( const std::vector<Tensor>& _validationSet, const std::vector<Tensor>& _validationSetExpectedOutput );

	const Layer* DbgGetLayer( uint32_t _idx ) const { return m_Layers[_idx].get(); }

private:
	void ClearWeightDeltas();
	void ApplyWeightDeltas( float _learningRate );
	void BackPropagation( const Tensor& _input, const Tensor& _expectedOutput );

private:
	std::vector< std::unique_ptr< Layer > > m_Layers;
};

uint32_t GetMostProbableClassIndex( const Tensor& _tensor );
void Dump( const Tensor& t );