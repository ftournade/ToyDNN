#pragma once

#include "Util.h"

#include <windows.h>
#include <vector>
#include <string>

namespace ToyDNN
{

	class Plot
	{
	public:
		enum Option
		{
			ShowXAxis = 1,
			ShowYAxis = 2
		};

		template< typename T >
		bool PlotCurve( const std::string& _name, const std::string& _xaxisLabel, const std::string& _yaxisLabel,
						const Color& _color, uint32_t _lineWidth,
						const std::vector<T>& x, const std::vector<T>& y );

		void Draw( CDC& _dc, const CRect& _rect, uint32_t _options = 0 );

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
}
