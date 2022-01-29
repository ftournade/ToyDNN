#pragma once
#include <windows.h>
#include "NeuralNetwork.h"
#include "Datasets.h"

using namespace ToyDNN;

class BaseExample
{
public:
	BaseExample();
	virtual ~BaseExample() {}

	void SetHwnd( HWND _hWnd ) { m_hWnd = _hWnd; }

	virtual void Tick( CDC& _dc ) = 0;

	virtual void OnLMouseButtonDown( const POINT& p ) { m_LMouseButtonDown = true; SetCapture( m_hWnd ); }
	virtual void OnLMouseButtonUp( const POINT& p ) { m_LMouseButtonDown = false; ReleaseCapture(); }
	virtual void OnRMouseButtonDown( const POINT& p ) { m_RMouseButtonDown = true; SetCapture( m_hWnd ); }
	virtual void OnRMouseButtonUp( const POINT& p ) { m_RMouseButtonDown = false; ReleaseCapture(); }
	virtual void OnMouseMove( const POINT& p ) {}

protected:
	inline bool IsLMouseButtonDown() const { return m_LMouseButtonDown; }

	void PlotLearningCurve( CDC& _dc, const CRect& _r ) const;

protected:
	HWND m_hWnd;
	NeuralNetwork net;

	CPen m_hBlackPen, m_hRedPen;

	bool m_LMouseButtonDown = false;
	bool m_RMouseButtonDown = false;
};

//Basic network, 2-2-2 fully connected
class Example1 : public BaseExample
{
public:
	Example1();
	virtual void Tick( CDC& _dc ) override;

protected:
	std::vector< Tensor > m_Input;
	std::vector< Tensor > m_ExpectedOutput;
};

//Curve fitting, 2-2-2 fully connected
class Example2 : public BaseExample
{
public:
	Example2();
	virtual void Tick( CDC& _dc ) override;

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
	virtual void Tick( CDC& _dc ) override;

	virtual void OnLMouseButtonDown( const POINT& p ) override;
	virtual void OnRMouseButtonDown( const POINT& p ) override;
	virtual void OnMouseMove( const POINT& p ) override;

private:
	void DrawConvolutionLayerFeatures( CDC& _dc, uint32_t _zoom=1 );
	void DrawUserDrawnDigit( CDC& _dc );
private:

	#ifdef USE_CIFAR10_INSTEAD_OF_MNIST
	static const uint32_t m_ImageRes = 32;
	#else
	static const uint32_t m_ImageRes = 28;
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

	const RECT m_UserDrawDigitRect = { 400, 400, 600, 600 };
	Tensor m_UserDrawnDigit;
	uint32_t m_RecognizedDigit = 0;
};

//Basic Encoder-decoder on celebA dataset
class Example4 : public BaseExample
{
public:
	Example4();
	virtual void Tick( CDC& _dc ) override;

private:
	std::vector< Tensor > m_TrainingData;
	std::vector< CelebAMetaData > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< CelebAMetaData > m_ValidationMetaData;
};