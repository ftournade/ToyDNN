#include "Examples.h"


void TransformCurve( const RECT& _r, float _xScale, float _yScale, int _epoch, float _error, int& _x, int& _y )
{
    _x = (int)(_r.left + _epoch * _xScale);
    _y = (int)(_r.bottom - _error * _yScale);
}

BaseExample::BaseExample()
{
    m_hBlackPen = CreatePen( PS_SOLID, 2, RGB( 0, 0, 0 ) );
    m_hRedPen = CreatePen( PS_SOLID, 3, RGB( 255, 0, 0 ) );
}

void BaseExample::PlotLearningCurve( HDC _hdc, const RECT& _r ) const
{
    if( m_LearningCurve.empty() )
        return;

    //Find bounds

    int minx = 100000000, maxx = -100000000;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for( uint32_t i = 0 ; i < m_LearningCurve.size() ; ++i )
    {
        const auto& p = m_LearningCurve[i];

        minx = min( minx, (int)p.first );
        maxx = max( maxx, (int)p.first );

        miny = min( miny, p.second.learningSetCost );
        miny = min( miny, p.second.testingSetCost );
        maxy = max( maxy, p.second.learningSetCost );
        maxy = max( maxy, p.second.testingSetCost );
    }

    float xScale = (float)(_r.right - _r.left) / (float)maxx;
    float yScale = (float)(_r.bottom - _r.top) / maxy;

    //Background
    SelectObject( _hdc, GetStockObject( DC_BRUSH ) );
    SetDCBrushColor( _hdc, RGB( 200, 200, 200 ) );
    
    Rectangle( _hdc, _r.left, _r.top, _r.right, _r.bottom );

    //Draw curve

    SelectObject( _hdc, m_hRedPen );

    int x, y;
    TransformCurve( _r, xScale, yScale, m_LearningCurve[0].first, m_LearningCurve[0].second.learningSetCost, x, y );
    MoveToEx( _hdc, x, y, NULL );
    
    for( uint32_t i = 1 ; i < m_LearningCurve.size() ; ++i )
    {
        TransformCurve( _r, xScale, yScale, m_LearningCurve[i].first, m_LearningCurve[i].second.learningSetCost, x, y );

        if( x > _r.right )
            break;

        LineTo( _hdc, x, y );
    }

    //Axes

    SelectObject( _hdc, m_hBlackPen );

    MoveToEx( _hdc, _r.left, _r.bottom, NULL );
    LineTo( _hdc, _r.right, _r.bottom );

    MoveToEx( _hdc, _r.left, _r.bottom, NULL );
    LineTo( _hdc, _r.left, _r.top );

}

Example1::Example1()
{
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( 2, 2 ) );

    m_ExpectedOutput.push_back( { 0.666f, 0.333f } );
    m_Input.push_back( { 0.9f, 0.2f } );
    
    Tensor out;
    net.Evaluate( m_Input[0], out );
    float initialError = NeuralNetwork::ComputeError( out, m_ExpectedOutput[0] );

    m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( 0, { initialError, initialError } ) );

}

void Example1::Tick( HDC _hdc )
{
    const int numEpochs = 10;

    float error = net.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput, numEpochs, 1, 0.5f );
     
    m_Epoch += numEpochs;

    m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( m_Epoch, { error, error } ) );

    RECT r{10,10,800,800};
    PlotLearningCurve( _hdc, r );
}

//-----------------------------------------

Example2::Example2()
{
    
    srand( 111 );
    //178x218 images

    if( !LoadCelebADataset( "D:\\Dev\\DeepLearning Datasets\\CelebA", 0.01f /*load 10% of database*/, 0.0001f, 
                            m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
        throw std::exception("Can't load celebA database");

    uint32_t numPixels = 178 * 218;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, numPixels / 4 ) );
    numPixels /= 4;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, numPixels / 4 ) );
    numPixels /= 4;
    net.AddLayer(std::make_unique<FullyConnectedLayer<Sigmoid>>(numPixels, numPixels / 4));
    numPixels /= 4;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, numPixels / 4 ) ); //Latent space (a.k.a. network "bottleneck")
    numPixels /= 4;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, numPixels * 4 ) );
    numPixels *= 4;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, numPixels * 4 ) );
    numPixels *= 4;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, numPixels * 4 ) );
    numPixels *= 4;
    net.AddLayer( std::make_unique<FullyConnectedLayer<Sigmoid>>( numPixels, 178 * 218 ) );
}

void Example2::Tick( HDC _hdc )
{
    net.Train( m_TrainingData, m_TrainingData, m_ValidationData, m_ValidationData, 1, 5, 0.02f );
}
