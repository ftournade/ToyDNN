#include "pch.h"
#include "Examples.h"

#include "Plot.h"


BaseExample::BaseExample() : m_hWnd( NULL )
{
    m_hBlackPen.CreatePen( PS_SOLID, 2, RGB( 0, 0, 0 ) );
    m_hRedPen.CreatePen( PS_SOLID, 3, RGB( 255, 0, 0 ) );
}


void BaseExample::PlotLearningCurve( CDC& _dc, const CRect& _r ) const
{
    Plot plot;
    plot.PlotCurve( "Training error", "x", "y", Color( 1, 0, 0 ), 1, net.GetHistory().TrainingSetErrorXAxis, net.GetHistory().TrainingSetError );
    plot.PlotCurve( "Validation error", "x", "y", Color( 0, 1, 0 ), 3, net.GetHistory().ValidationSetErrorXAxis, net.GetHistory().ValidationSetError );
    plot.Draw( _dc, _r, Plot::ShowXAxis );
}

Example1::Example1()
{
    net.AddLayer( new FullyConnected( 2 ) );
    net.AddLayer( new Sigmoid() );
    net.Compile( TensorShape( 2 ) );

    m_ExpectedOutput.push_back( { 0.666f, 0.333f } );
    m_Input.push_back( { 0.9f, 0.2f } );
}

void Example1::Tick( CDC& _dc )
{
    //Hyper parameters
    const int numEpochs = 100;
    const int batchSize = 1;
    const int validationInterval = 300;
    const float learningRate = 0.5;
    const float errorTarget = 0.01f;

    net.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput, numEpochs, batchSize, validationInterval, learningRate );
}

//-----------------------------------------

Example2::Example2()
{
    srand( 666 );

    net.AddLayer( new FullyConnected( 20 ) );
    net.AddLayer( new Sigmoid() );
    net.AddLayer( new FullyConnected( 1 ) );
    net.AddLayer( new Tanh() );
    net.Compile( TensorShape( 1 ) );

    const uint32_t numSamples = 400;
    const float rangeMin = -10.0f;
    const float rangeMax = 10.0f;
    const float step = (rangeMax - rangeMin) / (float)numSamples;

    m_GroundTruthXAxis.resize( numSamples );
    m_GroundTruthYAxis.resize( numSamples );
    m_Input.resize( numSamples );
    m_ExpectedOutput.resize( numSamples );

    for( uint32_t i = 0 ; i < numSamples ; ++i )
    {
        float x = rangeMin + step * (float)i;
       // float y = x * std::sin( x * std::abs( std::cos( x ) ) ); //The curve we want to fit
        float y = std::sin( x );

        m_Input[i].push_back( x );
        m_ExpectedOutput[i].push_back( y );
    
        m_GroundTruthXAxis[i] = x;
        m_GroundTruthYAxis[i] = y;
    }

}

void Example2::Tick( CDC& _dc )
{
    //Hyper parameters
    const int numEpochs = 1;
    const int batchSize = 100;
    const int validationInterval = 10;
    const float learningRate = 0.00004f;
    const float errorTarget = 0.01f;


    net.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput, numEpochs, batchSize, validationInterval, learningRate );


  //  net.GradientCheck( m_Input, m_ExpectedOutput, 10 );

    std::vector< Scalar > predictedCurve( m_GroundTruthXAxis.size() );

    for( uint32_t i = 0 ; i < m_GroundTruthXAxis.size() ; ++i )
    {
        Tensor in, out;
        in.push_back( m_GroundTruthXAxis[i] );
        net.Evaluate( in, out );
        predictedCurve[i] = out[0];
    }

    Plot plot;
    plot.PlotCurve( "", "x", "y", Color( 1, 0, 0 ), 4, m_GroundTruthXAxis, m_GroundTruthYAxis );
    plot.PlotCurve( "", "x", "y", Color( 0, 1, 0 ), 4, m_GroundTruthXAxis, predictedCurve );
    plot.Draw( _dc, CRect( 10, 10, 800, 800 ) );

    PlotLearningCurve( _dc, CRect( 10, 900, 800, 1200 ) );
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

#if 0
    int dbgSize = 1;

    m_DebugData.resize( dbgSize );
    m_DebugMetaData.resize( dbgSize );

    for( int i = 0; i < dbgSize ; ++i )
    {
        m_DebugData[i] = m_ValidationData[i];
        m_DebugMetaData[i] = m_ValidationMetaData[i];
    }
#endif
}

void Example3::Tick( CDC& _dc )
{
    //Hyper parameters
    const int numEpochs = 1;
    const int batchSize = 32;
    const int validationInterval = 100;
    const float learningRate = 0.001f;
    const float errorTarget = 0.02f;

    if( !m_IsTrained )
    {
        net.Train( m_TrainingData, m_TrainingMetaData, m_ValidationData, m_ValidationMetaData, numEpochs, batchSize, validationInterval, learningRate, errorTarget );

        //TODO if( error < errorTarget )
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

        TCHAR buffer[64];
        _stprintf_s( buffer, _T("Recognized digit: %d"), m_RecognizedDigit );
        _dc.DrawText( buffer, -1, &r, DT_CENTER );
    
        DrawUserDrawnDigit( _dc );
    
      //  DrawConvolutionLayerFeatures( _hdc, 3 ); //SLOW
    }

    PlotLearningCurve( _dc, CRect( 10, 400, 800, 800 ) );
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

void Example3::DrawConvolutionLayerFeatures( CDC& _dc, uint32_t _zoom )
{
    uint32_t s = (m_ImageRes - m_KernelSize + 1) / (m_Stride * m_Stride);

    const Tensor& convOutput = net.DbgGetLayer( 0 )->GetOutput();

    for( uint32_t f = 0 ; f < m_NumFeatureMaps ; ++f )
    {
        for( uint32_t y = 0 ; y < s ; ++y )
        {
            for( uint32_t x = 0 ; x < s ; ++x )
            {
                float v = (float)convOutput[ f * s * s + y * s + x ];
                v = v * 255.0f;
                v = std::min( v, 255.0f );
                v = std::max( v, 0.0f );

                DWORD col = RGB( (int)v, (int)v, (int)v );

                for( uint32_t dy = 0 ; dy < _zoom ; ++dy )
                {
                    for( uint32_t dx = 0 ; dx < _zoom ; ++dx )
                    {
                        _dc.SetPixel( x * _zoom + dx + f * (_zoom * s + 3), y * _zoom + dy, col );
                    }
                }

            }
        }
    }

}

void Example3::DrawUserDrawnDigit( CDC& _dc )
{
    _dc.SelectStockObject( WHITE_BRUSH );
    _dc.Rectangle( &m_UserDrawDigitRect );
    CBrush blackBrush;
    blackBrush.CreateStockObject( BLACK_BRUSH );
    _dc.FrameRect( &m_UserDrawDigitRect, &blackBrush );

    _dc.SelectStockObject( BLACK_BRUSH );

    uint32_t pixelWidth = (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left) / m_ImageRes;
    uint32_t pixelHeight = (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top) / m_ImageRes;

    for( uint32_t y = 0 ; y < m_ImageRes ; ++y )
    {
        for( uint32_t x = 0 ; x < m_ImageRes ; ++x )
        {
            if( m_UserDrawnDigit[y * m_ImageRes + x] > 0.0f )
            {
                _dc.Rectangle(    m_UserDrawDigitRect.left + x * pixelWidth, m_UserDrawDigitRect.top + y * pixelHeight,
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

void Example4::Tick( CDC& _dc )
{
    //Hyper parameters
    const int numEpochs = 100;
    const int batchSize = 30;
    const int validationInterval = 20;
    const float learningRate = 0.002f;
    const float errorTarget = 0.04f;

    net.Train( m_TrainingData, m_TrainingData, m_ValidationData, m_ValidationData, numEpochs, batchSize, validationInterval, learningRate, errorTarget );
}
