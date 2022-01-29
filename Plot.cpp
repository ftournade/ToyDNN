#include "pch.h"
#include "Plot.h"

#undef min
#undef max

void Plot::PlotCurve( Curve& _curve, const std::string& _name, const std::string& _xaxisLabel, const std::string& _yaxisLabel,
				const Color& _color, uint32_t _lineWidth )
{
	_curve.LineColor = _color;
	_curve.LineWidth = _lineWidth;
	_curve.Name = _name;
	_curve.XAxisLabel = _xaxisLabel;
	_curve.YAxisLabel = _yaxisLabel;
	_curve.XMin = FLT_MAX;
	_curve.XMax = -FLT_MAX;
	_curve.YMin = FLT_MAX;
	_curve.YMax = -FLT_MAX;

	for( const auto& pt : _curve.Points )
	{
		_curve.XMin = std::min( _curve.XMin, pt.first );
		_curve.XMax = std::max( _curve.XMax, pt.first );
		_curve.YMin = std::min( _curve.YMin, pt.second );
		_curve.YMax = std::max( _curve.YMax, pt.second );
	}
}

void Plot::Draw( CDC& _dc, const CRect& _rect )
{
	const int margin = 50;

	float XMin = FLT_MAX;
	float XMax = -FLT_MAX;
	float YMin = FLT_MAX;
	float YMax = -FLT_MAX;

	for( size_t i=0 ; i < m_Curves.size() ; ++i )
	{
		XMin = std::min( XMin, m_Curves[i].XMin );
		XMax = std::max( XMax, m_Curves[i].XMax );
		YMin = std::min( YMin, m_Curves[i].YMin );
		YMax = std::max( YMax, m_Curves[i].YMax );
	}

	CRect r = _rect;
	r.DeflateRect( margin, margin );
	
	//Draw background
	_dc.FillSolidRect( &_rect, RGB( 255, 255, 255 ) );
	
	//Draw axes
	_dc.SelectStockObject( DC_PEN );
	_dc.SetDCPenColor( RGB( 0, 0, 0 ) );

	_dc.MoveTo( r.left, r.bottom );
	_dc.LineTo( r.left, r.top );
	_dc.MoveTo( r.left, r.bottom );
	_dc.LineTo( r.right, r.bottom );

	//Draw curves
	float xScale = r.Width() / (XMax - XMin);
	float xBias = (float)r.left - (float)r.Width() * XMin / (XMax - XMin);
	float yScale = -r.Height() / (YMax - YMin);
	float yBias = (float)r.top - (float)r.Height() * YMin / (YMax - YMin);

	for( const auto& curve : m_Curves )
	{
		CPen curvePen( PS_SOLID, curve.LineWidth, curve.LineColor );
		_dc.SelectObject( curvePen );

		bool firstPoint = true;

		for( const auto& pt : curve.Points )
		{
			int x = (int)((float)pt.first * xScale + xBias);
			int y = (int)((float)pt.second * yScale + yBias);
		
			if( firstPoint )
			{
				firstPoint = false;
				_dc.MoveTo( x, y );
			}
			else
				_dc.LineTo( x, y );
		}

	}
}
