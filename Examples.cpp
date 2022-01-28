#include "Examples.h"


void TransformCurve( const RECT& _r, float _xScale, float _yScale, int _epoch, float _error, int& _x, int& _y )
{
    _x = (int)(_r.left + _epoch * _xScale);
    _y = (int)(_r.bottom - _error * _yScale);
}

BaseExample::BaseExample() : m_hWnd( NULL )
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

        minx = std::min( minx, (int)p.first );
        maxx = std::max( maxx, (int)p.first );

        miny = std::min( miny, p.second.learningSetCost );
        miny = std::min( miny, p.second.testingSetCost );
        maxy = std::max( maxy, p.second.learningSetCost );
        maxy = std::max( maxy, p.second.testingSetCost );
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
    net.AddLayer( new FullyConnected( 2 ) );
    net.AddLayer( new Sigmoid() );
    net.Compile( TensorShape( 2 ) );

    m_ExpectedOutput.push_back( { 0.666f, 0.333f } );
    m_Input.push_back( { 0.9f, 0.2f } );
    
    Tensor out;
    net.Evaluate( m_Input[0], out );
    float initialError = NeuralNetwork::ComputeError( out, m_ExpectedOutput[0] );

    m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( 0, { initialError, initialError } ) );

}

void Example1::Tick( HDC _hdc )
{
    //Hyper parameters
    const int numEpochs = 100;
    const int batchSize = 1;
    const int validationInterval = 300;
    const float learningRate = 0.5;
    const float errorTarget = 0.01f;


    float error = net.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput, numEpochs, batchSize, validationInterval, learningRate );
     
    m_Epoch += numEpochs;

    m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( m_Epoch, { error, error } ) );

    RECT r{10,10,800,800};
    PlotLearningCurve( _hdc, r );
}

//-----------------------------------------

Example2::Example2()
{
    net.AddLayer( new FullyConnected( 400 ) );
    net.AddLayer( new FullyConnected( 1 ) );
    net.AddLayer( new Sigmoid() );
    net.Compile( TensorShape( 1 ) );

    const uint32_t numSamples = 1000;
    const float rangeMin = -10.0f;
    const float rangeMax = 10.0f;
    const float step = (rangeMax - rangeMin) / (float)numSamples;

    m_Input.resize( numSamples );
    m_ExpectedOutput.resize( numSamples );

    for( uint32_t i = 0 ; i < numSamples ; ++i )
    {
        float x = rangeMin + step * (float)i;
       // float y = x * std::sin( x * std::abs( std::cos( x ) ) ); //The curve we want to fit
        float y = std::sin( x );

        m_Input[i].push_back( x );
        m_ExpectedOutput[i].push_back( y );
    }

/*
    Tensor out;
    net.Evaluate( m_Input[0], out );
    float initialError = NeuralNetwork::ComputeError( out, m_ExpectedOutput[0] );

    m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( 0, { initialError, initialError } ) );
    */
}

void Example2::Tick( HDC _hdc )
{
    //Hyper parameters
    const int numEpochs = 10000;
    const int batchSize = 1000;
    const int validationInterval = 1;
    const float learningRate = 0.00002f;
    const float errorTarget = 0.01f;


    float error = net.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput, numEpochs, batchSize, validationInterval, learningRate );

    m_Epoch += numEpochs;

    m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( m_Epoch, { error, error } ) );

    RECT r{ 10,10,800,800 };
    PlotLearningCurve( _hdc, r );
}

//-----------------------------------------


Example3::Example3()
{
    srand( 111 );

#ifdef USE_CIFAR10_INSTEAD_OF_MNIST
    uint32_t numChannels = 3;
#else
    uint32_t numChannels = 1;
#endif

    net.AddLayer( new Convolution2D( m_NumFeatureMaps, m_KernelSize, m_Stride ) );
    net.AddLayer( new Relu() );
    net.AddLayer( new MaxPooling( 2, 2 ) );
    net.AddLayer( new Convolution2D( m_NumFeatureMaps * 2, m_KernelSize, m_Stride ) );
    net.AddLayer( new Relu() );
    net.AddLayer( new MaxPooling( 2, 2 ) );
    net.AddLayer( new FullyConnected( 400 ) );
    net.AddLayer( new Relu() );
    net.AddLayer( new FullyConnected( 10 ) );
    net.AddLayer( new SoftMax() );
    net.Compile( TensorShape( m_ImageRes, m_ImageRes, numChannels ) );

#ifdef USE_CIFAR10_INSTEAD_OF_MNIST
    if( !LoadCifar10Dataset( "D:\\Dev\\DeepLearning Datasets\\cifar10",
                           m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
        throw std::exception( "Can't load Cifar10 database" );
#else
    if( !LoadMnistDataset( "D:\\Dev\\DeepLearning Datasets\\MNIST",
                           0.0f, 1.0f, 0, 0,
                           m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
        throw std::exception( "Can't load MNIST database" );
#endif

    m_UserDrawnDigit.resize( m_ImageRes * m_ImageRes, 0 );
}

void Example3::Tick( HDC _hdc )
{
    //Hyper parameters
    const int numEpochs = 1;
    const int batchSize = 32;
    const int validationInterval = 100;
    const float learningRate = 0.0001f;
    const float errorTarget = 0.02f;

    if( !m_IsTrained )
    {
        float error = net.Train( m_TrainingData, m_TrainingMetaData, m_ValidationData, m_ValidationMetaData, numEpochs, batchSize, validationInterval, learningRate, errorTarget );

        m_Epoch += numEpochs;

        m_LearningCurve.push_back( std::pair<uint32_t, LearningCurveData>( m_Epoch, { error, error } ) );

        if( error < errorTarget )
            m_IsTrained = true;
    }
    else
    {
        static uint32_t frame = 0;
        
        if( frame++ % 60 == 0 ) //Don't update every frame
        {
            Tensor out;
            net.Evaluate( m_UserDrawnDigit, out );
            m_RecognizedDigit = GetMostProbableClassIndex( out );
        }
        
        RECT r = { m_UserDrawDigitRect.left, m_UserDrawDigitRect.bottom, m_UserDrawDigitRect.right, m_UserDrawDigitRect.bottom + 30 };

        char buffer[64];
        sprintf_s( buffer, "Recognized digit: %d", m_RecognizedDigit );
        DrawTextA( _hdc, buffer, -1, &r, DT_CENTER );
    
        DrawUserDrawnDigit( _hdc );
    
      //  DrawConvolutionLayerFeatures( _hdc, 3 ); //SLOW
    }

    RECT r{ 10,10,300,300 };
  //  PlotLearningCurve( _hdc, r );

}

void Example3::OnLMouseButtonDown( const POINT& p )
{
    BaseExample::OnLMouseButtonDown( p );

    if( !PtInRect( &m_UserDrawDigitRect, p ) )
        return;

    int x = ( m_ImageRes * (p.x - m_UserDrawDigitRect.left)) / (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left);
    int y = ( m_ImageRes * (p.y - m_UserDrawDigitRect.top) ) / (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top);

    assert( x >= 0 && y >= 0 );
    assert( x < m_ImageRes&& y < m_ImageRes );

    m_UserDrawnDigit[y * m_ImageRes + x] = 1.0f;
}

void Example3::OnMouseMove( const POINT& p )
{
    if( !IsLMouseButtonDown() || !PtInRect( &m_UserDrawDigitRect, p ) )
        return;

    int x = (m_ImageRes * (p.x - m_UserDrawDigitRect.left)) / (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left);
    int y = (m_ImageRes * (p.y - m_UserDrawDigitRect.top)) / (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top);

    assert( x >= 0 && y >= 0 );
    assert( x < m_ImageRes&& y < m_ImageRes );

    m_UserDrawnDigit[y * m_ImageRes + x] = 1.0f;
}


void Example3::OnRMouseButtonDown( const POINT& p )
{
    std::fill( m_UserDrawnDigit.begin(), m_UserDrawnDigit.end(), 0.0f );
}

void Example3::DrawConvolutionLayerFeatures( HDC _hdc, uint32_t _zoom )
{
    uint32_t s = (m_ImageRes - m_KernelSize + 1) / (m_Stride * m_Stride);

    const Tensor& convOutput = net.DbgGetLayer( 0 )->GetOutput();

    for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
    {
        for( uint32_t y = 0 ; y < s ; ++y )
        {
            for( uint32_t x = 0 ; x < s ; ++x )
            {
                float v = convOutput[ f * s * s + y * s + x ];
                v = v * 255.0f;
                v = std::min( v, 255.0f );
                v = std::max( v, 0.0f );

                DWORD col = RGB( (int)v, (int)v, (int)v );

                for( uint32_t dy = 0 ; dy < _zoom ; ++dy )
                {
                    for( uint32_t dx = 0 ; dx < _zoom ; ++dx )
                    {
                        SetPixel( _hdc, x * _zoom + dx + f * (_zoom * s + 3), y * _zoom + dy, col );
                    }
                }

            }
        }
    }

}

void Example3::DrawUserDrawnDigit( HDC _hdc )
{
    SelectObject( _hdc, GetStockObject( WHITE_BRUSH ) );
    Rectangle( _hdc, m_UserDrawDigitRect.left, m_UserDrawDigitRect.top, m_UserDrawDigitRect.right, m_UserDrawDigitRect.bottom );

    FrameRect( _hdc, &m_UserDrawDigitRect, (HBRUSH)GetStockObject( BLACK_BRUSH ) );

    SelectObject( _hdc, GetStockObject( BLACK_BRUSH ) );

    uint32_t pixelWidth = (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left) / m_ImageRes;
    uint32_t pixelHeight = (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top) / m_ImageRes;

    for( uint32_t y = 0 ; y < m_ImageRes ; ++y )
    {
        for( uint32_t x = 0 ; x < m_ImageRes ; ++x )
        {
            if( m_UserDrawnDigit[y * m_ImageRes + x] > 0.0f )
            {
                Rectangle( _hdc,    m_UserDrawDigitRect.left + x * pixelWidth, m_UserDrawDigitRect.top + y * pixelHeight,
                                    m_UserDrawDigitRect.left + (x+1) * pixelWidth, m_UserDrawDigitRect.top + (y+1) * pixelHeight );
            }
        }

    }
}

//-----------------------------------------

Example4::Example4()
{
    
    srand( 111 );

    const bool halfRes = true;

    TensorShape inputShape( 178, 218, 3 );

    if( halfRes )
        inputShape = TensorShape( inputShape.m_SX / 2, inputShape.m_SY / 2, 3 );

//    net.AddLayer( new Convolution2D( 16, 3, 2 ) );
//    net.AddLayer( new Relu() );
    net.AddLayer( new FullyConnected( 15 ) );
    net.AddLayer( new Relu() );
    net.AddLayer( new FullyConnected( inputShape.Size() ) );
    net.AddLayer( new Sigmoid() );
    net.Compile( inputShape );

    if( !LoadCelebADataset( "D:\\Dev\\DeepLearning Datasets\\CelebA", halfRes, 2.0f, 0.02f,
                            m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
        throw std::exception("Can't load celebA database");
}

void Example4::Tick( HDC _hdc )
{
    //Hyper parameters
    const int numEpochs = 100;
    const int batchSize = 30;
    const int validationInterval = 20;
    const float learningRate = 0.002f;
    const float errorTarget = 0.04f;

    net.Train( m_TrainingData, m_TrainingData, m_ValidationData, m_ValidationData, numEpochs, batchSize, validationInterval, learningRate, errorTarget );
}
