#pragma once
#include <windows.h>
#include "NeuralNetwork.h"
#include "Datasets.h"

using namespace ToyDNN;

struct HyperParameters
{
	uint32_t BatchSize;
	uint32_t ValidationInterval;
	Scalar LearningRate;
};

class BaseExample
{
public:
	BaseExample();
	virtual ~BaseExample() {}

	void StopTraining() { m_NeuralNet.StopTraining(); }
	void PauseTraining() { 
		m_IsTrainingPaused = true; 
		m_NeuralNet.StopTraining();

		//Wait until it really stops
		while( m_NeuralNet.IsTraining() )
		{
			Sleep( 20 );
		}
	}

	void ResumeTraining() { m_IsTrainingPaused = false; }

	void SetHwnd( HWND _hWnd ) { m_hWnd = _hWnd; }

	virtual void TrainingThread( const HyperParameters& _params ) = 0;
	virtual void Draw( CDC& _dc ) = 0;

	virtual void OnLMouseButtonDown( const POINT& p ) { m_LMouseButtonDown = true; SetCapture( m_hWnd ); }
	virtual void OnLMouseButtonUp( const POINT& p ) { m_LMouseButtonDown = false; ReleaseCapture(); }
	virtual void OnRMouseButtonDown( const POINT& p ) { m_RMouseButtonDown = true; SetCapture( m_hWnd ); }
	virtual void OnRMouseButtonUp( const POINT& p ) { m_RMouseButtonDown = false; ReleaseCapture(); }
	virtual void OnMouseMove( const POINT& p ) {}

protected:
	inline bool IsLMouseButtonDown() const { return m_LMouseButtonDown; }

	void PlotLearningCurve( CDC& _dc, const CRect& _r ) const;

protected:
	HWND m_hWnd = 0;
	NeuralNetwork m_NeuralNet;
	bool m_IsTrainingPaused = false;

	CPen m_hBlackPen, m_hRedPen;

	bool m_LMouseButtonDown = false;
	bool m_RMouseButtonDown = false;
};

//Basic network, 2-2-2 fully connected
class Example1 : public BaseExample
{
public:
	Example1();
	virtual void Draw( CDC& _dc ) override;

protected:
	std::vector< Tensor > m_Input;
	std::vector< Tensor > m_ExpectedOutput;
};

//Curve fitting, 2-2-2 fully connected
class Example2 : public BaseExample
{
public:
	Example2();
	virtual void TrainingThread( const HyperParameters& _params );
	virtual void Draw( CDC& _dc ) override;

protected:
	std::vector< Tensor > m_Input;
	std::vector< Tensor > m_ExpectedOutput;

	std::vector< Scalar > m_GroundTruthXAxis, m_GroundTruthYAxis;
};


//#define USE_CIFAR10_INSTEAD_OF_MNIST
//Basic MNIST classifier
class Example3 : public BaseExample
{
public:
	Example3();
	virtual void Draw( CDC& _dc ) override;

	virtual void OnLMouseButtonDown( const POINT& p ) override;
	virtual void OnRMouseButtonDown( const POINT& p ) override;
	virtual void OnMouseMove( const POINT& p ) override;

private:
	void DrawConvolutionLayerFeatures( CDC& _dc, uint32_t _zoom=1 );
	void DrawUserDrawnDigit( CDC& _dc );
private:

	#ifdef USE_CIFAR10_INSTEAD_OF_MNIST
	static const uint32_t m_ImageRes = 32;
	const char* m_NeuralNetFilename = "D:/tmp/example3_cifar10.dnn";
	#else
	static const uint32_t m_ImageRes = 28;
	const char* m_NeuralNetFilename = "D:/tmp/example3_mnist.dnn";
	#endif
	static const uint32_t m_NumFeatureMaps = 8;
	static const uint32_t m_KernelSize = 3;
	static const uint32_t m_Stride = 1;

	bool m_IsTrained = false;

	std::vector< Tensor > m_TrainingData;
	std::vector< Tensor > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< Tensor > m_ValidationMetaData;
	std::vector< Tensor > m_DebugData;
	std::vector< Tensor > m_DebugMetaData;

	const RECT m_UserDrawDigitRect = { 900, 400, 1300, 800 };
	Tensor m_UserDrawnDigit;
	uint32_t m_RecognizedDigit = 0;
};

//Basic Encoder-decoder on celebA dataset
class Example4 : public BaseExample
{
public:
	Example4();
	virtual void Draw( CDC& _dc ) override;

private:
	std::vector< Tensor > m_TrainingData;
	std::vector< CelebAMetaData > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< CelebAMetaData > m_ValidationMetaData;
};