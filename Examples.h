#pragma once
#include <windows.h>
#include "NeuralNetwork.h"
#include "Datasets.h"

class BaseExample
{
public:
	BaseExample();
	virtual ~BaseExample() {}
	virtual void Tick( HDC _hdc ) = 0;

protected:
	void PlotLearningCurve( HDC _hdc, const RECT& _r ) const;

protected:
	NeuralNetwork net;

	struct LearningCurveData
	{
		float learningSetCost; //train-set
		float testingSetCost; //testing-set: data never seen by the optimizer
	};

	std::vector< std::pair< uint32_t, LearningCurveData > > m_LearningCurve;
	
	uint32_t m_Epoch = 0;

	HPEN m_hBlackPen, m_hRedPen;
};

//Basic network, 2-2-2 fully connected
class Example1 : public BaseExample
{
public:
	Example1();
	virtual void Tick( HDC _hdc ) override;

private:
	std::vector< Tensor > m_Input;
	std::vector< Tensor > m_ExpectedOutput;
};

//Basic Encoder-decoder on celebA dataset
class Example2 : public BaseExample
{
public:
	Example2();
	virtual void Tick( HDC _hdc ) override;

private:
	std::vector< Tensor > m_TrainingData;
	std::vector< CelebAMetaData > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< CelebAMetaData > m_ValidationMetaData;
};