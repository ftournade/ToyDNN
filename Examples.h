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
	virtual ~BaseExample();

	void StopTraining( bool _waitForTrainingToStop = false );
	void PauseTraining();
	void ResumeTraining();

	void SetHwnd( HWND _hWnd ) { m_hWnd = _hWnd; }

	virtual void Train( const HyperParameters& _params ) = 0;	
	void TrainingThread( const HyperParameters& _params );
	
	virtual void Draw( CDC& _dc ) = 0;

	virtual void GradientCheck() {}

	//Return true if you want the view redrawn
	virtual bool OnLMouseButtonDown( const CPoint& p ) { m_LMouseButtonDown = true; SetCapture( m_hWnd ); return false; }
	virtual bool OnLMouseButtonUp( const CPoint& p ) { m_LMouseButtonDown = false; ReleaseCapture(); return false; }
	virtual bool OnRMouseButtonDown( const CPoint& p ) { m_RMouseButtonDown = true; SetCapture( m_hWnd ); return false; }
	virtual bool OnRMouseButtonUp( const CPoint& p ) { m_RMouseButtonDown = false; ReleaseCapture(); return false; }
	virtual bool OnMouseMove( const CPoint& p ) { return false; }

protected:
	inline bool IsLMouseButtonDown() const { return m_LMouseButtonDown; }

	void PlotLearningCurve( CDC& _dc, const CRect& _r ) const;
	void DrawConvolutionLayerFeatures( CDC& _dc, uint32_t _layerIndex, int _x, int _y, uint32_t _zoom=1 );

protected:
	HWND m_hWnd = 0;
	NeuralNetwork m_NeuralNet;
	bool m_IsTrainingPaused = false;
	bool m_StopTraining = false;

	bool m_LMouseButtonDown = false;
	bool m_RMouseButtonDown = false;
};

//Basic network, 2-2-2 fully connected
class Example1 : public BaseExample
{
public:
	Example1();
	virtual ~Example1() {}

	virtual void Train( const HyperParameters& _params ) override;
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
	virtual ~Example2() {}

	virtual void Train( const HyperParameters& _params ) override;
	virtual void GradientCheck() override;
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
	virtual ~Example3() {}

	virtual void Train( const HyperParameters& _params ) override;
	virtual void GradientCheck() override;
	virtual void Draw( CDC& _dc ) override;

	virtual bool OnLMouseButtonDown( const CPoint& p ) override;
	virtual bool OnRMouseButtonDown( const CPoint& p ) override;
	virtual bool OnMouseMove( const CPoint& p ) override;

private:
	void DrawUserDrawnDigit( CDC& _dc );
private:

	#ifdef USE_CIFAR10_INSTEAD_OF_MNIST
	static const uint32_t m_ImageRes = 32;
	const char* m_NeuralNetFilename = "D:/tmp/example3_cifar10.dnn";
	#else
	static const uint32_t m_ImageRes = 28;
	const char* m_NeuralNetFilename = "D:/tmp/example3_mnist.dnn";
	#endif
	static const uint32_t m_NumFeatureMaps = 16;
	static const uint32_t m_KernelSize = 3;
	static const uint32_t m_Stride = 1;

	bool m_IsTrained = false;

	std::vector< Tensor > m_TrainingData;
	std::vector< Tensor > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< Tensor > m_ValidationMetaData;
	std::vector< Tensor > m_DebugData;
	std::vector< Tensor > m_DebugMetaData;

	const CRect m_UserDrawDigitRect = { 900, 400, 1300, 800 };
	Tensor m_UserDrawnDigit;
	uint32_t m_RecognizedDigit = 0;
};

//Basic Encoder-decoder on celebA dataset
class Example4 : public BaseExample
{
public:
	Example4();
	virtual ~Example4() {}

	virtual void Train( const HyperParameters& _params ) override;
	virtual void Draw( CDC& _dc ) override;

private:
	std::vector< Tensor > m_TrainingData;
	std::vector< CelebAMetaData > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< CelebAMetaData > m_ValidationMetaData;
};