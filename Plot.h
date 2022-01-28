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
	bool PlotCurve( const std::string& _name, const std::string& _xaxisLabel, const std::string& _yaxisLabel,
					const Color& _color, uint32_t _lineWidth,
					const std::vector<float>& x, const std::vector<float>& y );
	void Draw( CDC& _dc, const CRect& _rect );

protected:
	
private:
	struct Curve
	{
		Color LineColor;
		uint32_t LineWidth;
		std::string Name, XAxisLabel, YAxisLabel;
		std::vector<std::pair<float, float>> Points;
	
		float XMin, XMax, YMin, YMax;
	};
		
	std::vector<Curve> m_Curves;
};

