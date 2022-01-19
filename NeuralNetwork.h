#pragma once

#include "Layer.h"

#include <memory>

class NeuralNetwork
{
public:
	void AddLayer( std::unique_ptr<Layer> _layer ) { m_Layers.push_back( std::move(_layer) ); }

	//Return error metric
	float Train( const std::vector<Tensor>& _trainingSet,
				 const std::vector<Tensor>& _trainingSetExpectedOutput,
				 const std::vector<Tensor>& _validationSet,
				 const std::vector<Tensor>& _validationSetExpectedOutput,
				 uint32_t _numEpochs, uint32_t _batchSize, float _learningRate );
	void Evaluate( const Tensor& _in, Tensor& _out ) const;

	static void ComputeError( const Tensor& _out, const Tensor& _expectedOutput, Tensor& _error );
	static float ComputeError( const Tensor& _out, const Tensor& _expectedOutput );
	float ComputeError( const std::vector<Tensor>& _validationSet, const std::vector<Tensor>& _validationSetExpectedOutput );

private:
	void ClearWeightDeltas();
	void ApplyWeightDeltas( float _learningRate );
	void BackPropagation( const Tensor& _input, const Tensor& _expectedOutput );

private:
	std::vector< std::unique_ptr< Layer > > m_Layers;
};

void Dump( const Tensor& t );