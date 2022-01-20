#pragma once
#include <windows.h>
#include "NeuralNetwork.h"
#include "Datasets.h"

class BaseExample
{
public:
	BaseExample();
	virtual ~BaseExample() {}

	void SetHwnd( HWND _hWnd ) { m_hWnd = _hWnd; }

	virtual void Tick( HDC _hdc ) = 0;

	virtual void OnLMouseButtonDown( const POINT& p ) { m_LMouseButtonDown = true; SetCapture( m_hWnd ); }
	virtual void OnLMouseButtonUp( const POINT& p ) { m_LMouseButtonDown = false; ReleaseCapture(); }
	virtual void OnRMouseButtonDown( const POINT& p ) { m_RMouseButtonDown = true; SetCapture( m_hWnd ); }
	virtual void OnRMouseButtonUp( const POINT& p ) { m_RMouseButtonDown = false; ReleaseCapture(); }
	virtual void OnMouseMove( const POINT& p ) {}

protected:
	inline bool IsLMouseButtonDown() const { return m_LMouseButtonDown; }

	void PlotLearningCurve( HDC _hdc, const RECT& _r ) const;

protected:
	HWND m_hWnd;
	NeuralNetwork net;

	struct LearningCurveData
	{
		float learningSetCost; //train-set
		float testingSetCost; //testing-set: data never seen by the optimizer
	};

	std::vector< std::pair< uint32_t, LearningCurveData > > m_LearningCurve;
	
	uint32_t m_Epoch = 0;

	HPEN m_hBlackPen, m_hRedPen;

	bool m_LMouseButtonDown = false;
	bool m_RMouseButtonDown = false;
};

//Basic network, 2-2-2 fully connected
class Example1 : public BaseExample
{
public:
	Example1();
	virtual void Tick( HDC _hdc ) override;

protected:
	std::vector< Tensor > m_Input;
	std::vector< Tensor > m_ExpectedOutput;
};

//Basic MNIST classifier
class Example3 : public BaseExample
{
public:
	Example3();
	virtual void Tick( HDC _hdc ) override;

	virtual void OnLMouseButtonDown( const POINT& p ) override;
	virtual void OnRMouseButtonDown( const POINT& p ) override;
	virtual void OnMouseMove( const POINT& p ) override;

private:

	void DrawUserDrawnDigit( HDC _dc );
private:

	static const uint32_t m_ImageRes = 28;

	bool m_IsTrained = false;

	std::vector< Tensor > m_TrainingData;
	std::vector< Tensor > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< Tensor > m_ValidationMetaData;

	const RECT m_UserDrawDigitRect = { 400, 400, 600, 600 };
	Tensor m_UserDrawnDigit;
};

//Basic Encoder-decoder on celebA dataset
class Example4 : public BaseExample
{
public:
	Example4();
	virtual void Tick( HDC _hdc ) override;

private:
	std::vector< Tensor > m_TrainingData;
	std::vector< CelebAMetaData > m_TrainingMetaData;
	std::vector< Tensor > m_ValidationData;
	std::vector< CelebAMetaData > m_ValidationMetaData;
};