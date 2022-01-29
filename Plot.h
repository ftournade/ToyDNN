#pragma once

#include <windows.h>
#include <vector>
#include <string>

class Color
{
public:
	Color() {}
	Color( float r, float g, float b ) : R(r), G(g), B(b) {}

	inline operator COLORREF() const { return RGB((int)(R*255.0f), (int)(G*255.0f), (int)(B*255.0f) ); }

	float R, G, B;
};

class Plot
{
public:
	template< typename T >
	bool PlotCurve( const std::string& _name, const std::string& _xaxisLabel, const std::string& _yaxisLabel,
					const Color& _color, uint32_t _lineWidth,
					const std::vector<T>& x, const std::vector<T>& y );
	void Draw( CDC& _dc, const CRect& _rect );


private:
	struct Curve
	{
		Color LineColor;
		uint32_t LineWidth;
		std::string Name, XAxisLabel, YAxisLabel;
		std::vector<std::pair<float, float>> Points;
	
		float XMin, XMax, YMin, YMax;
	};

	void PlotCurve( Curve& _outCurve, const std::string& _name, const std::string& _xaxisLabel, const std::string& _yaxisLabel,
					const Color& _color, uint32_t _lineWidth );

private:
	std::vector<Curve> m_Curves;
};


template< typename T >
bool Plot::PlotCurve( const std::string& _name, const std::string& _xaxisLabel, const std::string& _yaxisLabel,
					  const Color& _color, uint32_t _lineWidth,
					  const std::vector<T>& _x, const std::vector<T>& _y )
{
	if( _x.size() != _y.size() )
		return false;

	Curve curve;
	curve.Points.resize( _x.size() );

	for( size_t i = 0 ; i < _x.size() ; ++i )
	{
		curve.Points[i] = std::make_pair( (float)_x[i], (float)_y[i] );
	}

	PlotCurve( curve, _name, _xaxisLabel, _yaxisLabel, _color, _lineWidth );

	m_Curves.emplace_back( curve );

	return true;
}
