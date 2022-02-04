#pragma once

#include "Layers/Activation/ActivationLayers.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/Convolution2DLayer.h"
#include "Layers/MaxPoolingLayer.h"

#include <memory>

namespace ToyDNN
{

	class NeuralNetwork
	{
	public:
		void AddLayer( Layer* _layer );

		void Compile( const TensorShape& _inputShape );

		//Return error metric
		void Train(	 const std::vector<Tensor>& _trainingSet,
					 const std::vector<Tensor>& _trainingSetExpectedOutput,
					 const std::vector<Tensor>& _validationSet,
					 const std::vector<Tensor>& _validationSetExpectedOutput,
					 uint32_t _numEpochs, uint32_t _batchSize, uint32_t _validationInterval /*evaluate vaildationSet every N batch*/,
					 Scalar _learningRate, Scalar _errorTarget = 0.0001f );
		
		void StopTraining() { m_StopTraining = true; }
		bool IsTraining() { return m_IsTraining; }

		void Evaluate( const Tensor& _in, Tensor& _out ) const;

		static void ComputeError( const Tensor& _out, const Tensor& _expectedOutput, Tensor& _error );
		static Scalar ComputeError( const Tensor& _out, const Tensor& _expectedOutput );
		Scalar ComputeError( const std::vector<Tensor>& _validationSet, const std::vector<Tensor>& _validationSetExpectedOutput );

		void EnableClassificationAccuracyLog() { m_EnableClassificationAccuracyLog = true; }

		const Layer* DbgGetLayer( uint32_t _idx ) const { return m_Layers[_idx].get(); }
		uint32_t DbgGetLayerCount() const { return (uint32_t)m_Layers.size(); }

		//Used to debug gradient computation
		//The idea is to compare gradients computed during back propagation with gradients computed by finite difference.
		//We select random parameters (weights and biases) and we slightly perturb them one at a time, we then see the effect on the loss function.
		//Gradient = (EvalNetworkLoss( Param + epsilon ) - EvalNetworkLoss( Param - epsilon )) / (2 * epsilon)
		void GradientCheck( const std::vector<Tensor>& _dataSet, const std::vector<Tensor>& _dataSetExpectedOutput, uint32_t _numRandomParametersToCheck );

		void ClearHistory();

		struct History
		{
			std::vector<Scalar> TrainingSetErrorXAxis; //Epoch
			std::vector<Scalar> TrainingSetError;
			std::vector<Scalar> ValidationSetErrorXAxis; //Epoch
			std::vector<Scalar> ValidationSetError;
			
			uint32_t NumEpochCompleted = 0;
			uint32_t NumSamplesCompleted = 0;
			float CurrentAccuracy = 0.0f;
			float BestAccuracy = 0.0f;
		};

		const History& GetHistory() const { return m_History; }

		bool Load( const char* _filename );
		bool Save( const char* _filename, bool _saveTrainingHistory = false ) const;

	private:
		void ClearWeightDeltas();
		void ApplyWeightDeltas( Scalar _learningRate );
		void BackPropagation( const Tensor& _input, const Tensor& _expectedOutput );

	private:
		std::vector< std::unique_ptr<Layer> > m_Layers;
		
		History m_History;

		bool m_EnableClassificationAccuracyLog = false;
		bool m_StopTraining = false;
		bool m_IsTraining = false;
	};

	uint32_t GetMostProbableClassIndex( const Tensor& _tensor );
	void Dump( const Tensor& t );
}
