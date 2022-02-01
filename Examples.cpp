#include "pch.h"
#include "Examples.h"

#include "Plot.h"


BaseExample::BaseExample()
{
}


BaseExample::~BaseExample()
{
}

void BaseExample::StopTraining()
{
    m_NeuralNet.StopTraining();
}

void BaseExample::PauseTraining()
{
    m_IsTrainingPaused = true;
    m_NeuralNet.StopTraining();

    //Wait until it really stops
    while( m_NeuralNet.IsTraining() )
    {
        Sleep( 20 );
    }
}

void BaseExample::ResumeTraining()
{
    m_IsTrainingPaused = false; 
}

void BaseExample::TrainingThread( const HyperParameters& _params )
{

    while( true )
    {
        while( m_IsTrainingPaused )
        {
            //Pause training while the UI is drawn and potentially accesses the neural net
            //We could have used a mutex instead, but it is cheaper that way
            Sleep( 100 );
        }

        Train( _params );

        if( !m_IsTrainingPaused )
            return;
    }
}


void BaseExample::PlotLearningCurve( CDC& _dc, const CRect& _r ) const
{
    Plot plot;
    plot.PlotCurve( "Training error", "x", "y", Color( 1, 0, 0 ), 1, m_NeuralNet.GetHistory().TrainingSetErrorXAxis, m_NeuralNet.GetHistory().TrainingSetError );
    plot.PlotCurve( "Validation error", "x", "y", Color( 0, 1, 0 ), 3, m_NeuralNet.GetHistory().ValidationSetErrorXAxis, m_NeuralNet.GetHistory().ValidationSetError );
    plot.Draw( _dc, _r, Plot::ShowXAxis );
}

//=========================================

Example1::Example1()
{
    m_NeuralNet.AddLayer( new FullyConnected( 2 ) );
    m_NeuralNet.AddLayer( new Sigmoid() );
    m_NeuralNet.Compile( TensorShape( 2 ) );

    m_ExpectedOutput.push_back( { 0.666f, 0.333f } );
    m_Input.push_back( { 0.9f, 0.2f } );
}

void Example1::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_NeuralNet.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput,
                       numEpochs, _params.BatchSize, _params.ValidationInterval, _params.LearningRate );
}

void Example1::Draw( CDC& _dc )
{
    //Hyper parameters
    const int numEpochs = 100;
    const int batchSize = 1;
    const int validationInterval = 300;
    const float learningRate = 0.5;
    const float errorTarget = 0.01f;

    m_NeuralNet.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput, numEpochs, batchSize, validationInterval, learningRate );
}

//-----------------------------------------

Example2::Example2()
{
    srand( 666 );

    m_NeuralNet.AddLayer( new FullyConnected( 100 ) );
    m_NeuralNet.AddLayer( new LeakyRelu() );
    m_NeuralNet.AddLayer( new FullyConnected( 100 ) );
    m_NeuralNet.AddLayer( new LeakyRelu() );
    m_NeuralNet.AddLayer( new FullyConnected( 1 ) );
    m_NeuralNet.AddLayer( new Tanh() );
    m_NeuralNet.Compile( TensorShape( 1 ) );

    const uint32_t numSamples = 1000;
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
       // float y = std::sin( x * std::abs( std::cos( x ) ) ); //The curve we want to fit
        float y = std::sin( x );

        m_Input[i].push_back( x / 20.0f );
        m_ExpectedOutput[i].push_back( y );
    
        m_GroundTruthXAxis[i] = x / 20.0f;
        m_GroundTruthYAxis[i] = y;
    }

}

void Example2::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_NeuralNet.Train( m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput,
                       numEpochs, _params.BatchSize, _params.ValidationInterval, _params.LearningRate );
}

void Example2::Draw( CDC& _dc )
{
  //  m_NeuralNet.GradientCheck( m_Input, m_ExpectedOutput, 10 );

    std::vector< Scalar > predictedCurve( m_GroundTruthXAxis.size() );

    for( uint32_t i = 0 ; i < m_GroundTruthXAxis.size() ; ++i )
    {
        Tensor in, out;
        in.push_back( m_GroundTruthXAxis[i] );
        m_NeuralNet.Evaluate( in, out );
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

    if( m_NeuralNet.Load( m_NeuralNetFilename ) )
    {
        m_IsTrained = true;
    }
    else
    {
        m_NeuralNet.AddLayer( new Convolution2D( m_NumFeatureMaps, m_KernelSize, m_Stride ) );
        m_NeuralNet.AddLayer( new Relu() );
        m_NeuralNet.AddLayer( new MaxPooling( 2, 2 ) );
        m_NeuralNet.AddLayer( new FullyConnected( 100 ) );
        m_NeuralNet.AddLayer( new Relu() );
        m_NeuralNet.AddLayer( new FullyConnected( 10 ) );
        m_NeuralNet.AddLayer( new Sigmoid() );
        m_NeuralNet.Compile( TensorShape( m_ImageRes, m_ImageRes, numChannels ) );
        m_NeuralNet.EnableClassificationAccuracyLog();

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
    }
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

void Example3::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_NeuralNet.Train( m_TrainingData, m_TrainingMetaData, m_ValidationData, m_ValidationMetaData,
                       numEpochs, _params.BatchSize, _params.ValidationInterval, _params.LearningRate );
}

void Example3::Draw( CDC& _dc )
{
    Tensor out;
    m_NeuralNet.Evaluate( m_UserDrawnDigit, out );
    m_RecognizedDigit = GetMostProbableClassIndex( out );
        
    RECT r = { m_UserDrawDigitRect.left, m_UserDrawDigitRect.bottom, m_UserDrawDigitRect.right, m_UserDrawDigitRect.bottom + 30 };

    TCHAR buffer[64];
    _stprintf_s( buffer, _T("Recognized digit: %d"), m_RecognizedDigit );
    _dc.DrawText( buffer, -1, &r, DT_CENTER );
    
    DrawUserDrawnDigit( _dc );
    
    //  DrawConvolutionLayerFeatures( _hdc, 3 ); //SLOW

    PlotLearningCurve( _dc, CRect( 10, 400, 800, 800 ) );
}

bool Example3::OnLMouseButtonDown( const CPoint& p )
{
    BaseExample::OnLMouseButtonDown( p );

    if( !PtInRect( &m_UserDrawDigitRect, p ) )
        return false;

    int x = ( m_ImageRes * (p.x - m_UserDrawDigitRect.left)) / (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left);
    int y = ( m_ImageRes * (p.y - m_UserDrawDigitRect.top) ) / (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top);

    assert( x >= 0 && y >= 0 );
    assert( x < m_ImageRes&& y < m_ImageRes );

    m_UserDrawnDigit[y * m_ImageRes + x] = 1.0f;

    return true;
}

bool Example3::OnMouseMove( const CPoint& p )
{
    if( !IsLMouseButtonDown() || !m_UserDrawDigitRect.PtInRect( p ) )
        return false;

    int x = (m_ImageRes * (p.x - m_UserDrawDigitRect.left)) / (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left);
    int y = (m_ImageRes * (p.y - m_UserDrawDigitRect.top)) / (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top);

    assert( x >= 0 && y >= 0 );
    assert( x < m_ImageRes&& y < m_ImageRes );

    m_UserDrawnDigit[y * m_ImageRes + x] = 1.0f;
    return true;
}


bool Example3::OnRMouseButtonDown( const CPoint& p )
{
    std::fill( m_UserDrawnDigit.begin(), m_UserDrawnDigit.end(), 0.0f );
    return true;
}

void Example3::DrawConvolutionLayerFeatures( CDC& _dc, uint32_t _zoom )
{
    uint32_t s = (m_ImageRes - m_KernelSize + 1) / (m_Stride * m_Stride);

    const Tensor& convOutput = m_NeuralNet.DbgGetLayer( 0 )->GetOutput();

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
    _dc.FillSolidRect( &m_UserDrawDigitRect, RGB(255,255,255) );

    uint32_t pixelWidth = (m_UserDrawDigitRect.right - m_UserDrawDigitRect.left) / m_ImageRes;
    uint32_t pixelHeight = (m_UserDrawDigitRect.bottom - m_UserDrawDigitRect.top) / m_ImageRes;

    for( uint32_t y = 0 ; y < m_ImageRes ; ++y )
    {
        for( uint32_t x = 0 ; x < m_ImageRes ; ++x )
        {
            if( m_UserDrawnDigit[y * m_ImageRes + x] > 0.0f )
            {
                CRect pixel( m_UserDrawDigitRect.left + x * pixelWidth, m_UserDrawDigitRect.top + y * pixelHeight,
                             m_UserDrawDigitRect.left + (x+1) * pixelWidth, m_UserDrawDigitRect.top + (y+1) * pixelHeight );
            
                _dc.FillSolidRect( &pixel, RGB( 0, 0, 0 ) );
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

//    m_NeuralNet.AddLayer( new Convolution2D( 16, 3, 2 ) );
//    m_NeuralNet.AddLayer( new Relu() );
    m_NeuralNet.AddLayer( new FullyConnected( 15 ) );
    m_NeuralNet.AddLayer( new Relu() );
    m_NeuralNet.AddLayer( new FullyConnected( inputShape.Size() ) );
    m_NeuralNet.AddLayer( new Sigmoid() );
    m_NeuralNet.Compile( inputShape );

    if( !LoadCelebADataset( "D:\\Dev\\DeepLearning Datasets\\CelebA", halfRes, 2.0f, 0.02f,
                            m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
        throw std::exception("Can't load celebA database");
}

void Example4::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_NeuralNet.Train( m_TrainingData, m_TrainingData, m_ValidationData, m_ValidationData,
                       numEpochs, _params.BatchSize, _params.ValidationInterval, _params.LearningRate );
}

void Example4::Draw( CDC& _dc )
{
    PlotLearningCurve( _dc, CRect( 10, 400, 800, 800 ) );
}
