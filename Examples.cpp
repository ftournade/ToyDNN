#include "pch.h"
#include "Examples.h"

#include "Plot.h"


BaseExample::BaseExample()
{
}


BaseExample::~BaseExample()
{
}

void BaseExample::StopTraining( bool _waitForTrainingToStop )
{
    Log( "BaseExample::StopTraining\n" );

    m_StopTraining = true;

    m_NeuralNet.StopTraining();

    if( _waitForTrainingToStop )
    {
        while( m_NeuralNet.IsTraining() )
        {
            Sleep( 20 );
        }
    }
}

void BaseExample::PauseTraining()
{
    Log( "BaseExample::PauseTraining\n" );

    m_IsTrainingPaused = true;

    m_NeuralNet.StopTraining();

    //if( _waitForTrainingToStop )
    {
        while( m_NeuralNet.IsTraining() )
        {
            Sleep( 20 );
        }
    }
}

void BaseExample::ResumeTraining()
{
    Log( "BaseExample::ResumeTraining\n" );

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

        if( m_StopTraining )
        {
            m_StopTraining = false;
            return;
        }
    }
}


void BaseExample::PlotLearningCurve( CDC& _dc, const CRect& _r ) const
{
    Plot plot;
    plot.PlotCurve( "Training error", "x", "y", Color( 1, 0, 0 ), 1, m_NeuralNet.GetHistory().TrainingSetErrorXAxis, m_NeuralNet.GetHistory().TrainingSetError );
    plot.PlotCurve( "Validation error", "x", "y", Color( 0, 1, 0 ), 3, m_NeuralNet.GetHistory().ValidationSetErrorXAxis, m_NeuralNet.GetHistory().ValidationSetError );
    plot.Draw( _dc, _r, Plot::ShowXAxis );
}

void BaseExample::DrawConvolutionLayerFeatures( CDC& _dc, uint32_t _layerIndex, int _x, int _y, uint32_t _zoom )
{
    //assert( m_NeuralNet.DbgGetLayer( _layerIndex )->GetType() == LayerType::Convolution2D );

    const TensorShape& convOutputShape = m_NeuralNet.DbgGetLayer( _layerIndex )->GetOutputShape();
    const Tensor& convOutput = m_NeuralNet.DbgGetLayer( _layerIndex )->GetOutput();

    if( convOutput.empty() )
        return;

    uint32_t sx = convOutputShape.m_SX;
    uint32_t sy = convOutputShape.m_SY;

    for( uint32_t f = 0 ; f < convOutputShape.m_SZ ; ++f )
    {
    #if 1
        float vmin = FLT_MAX;
        float vmax = -FLT_MAX;

        for( uint32_t y = 0 ; y < sy ; ++y )
        {
            for( uint32_t x = 0 ; x < sx ; ++x )
            {
                float v = (float)convOutput[f * sx * sx + y * sx + x];

                vmin = std::min( vmin, v );
                vmax = std::max( vmax, v );
            }
        }

        float scale = 1.0f / (vmax - vmin);
    #else
        float scale = 1.0f;
        float vmin = 0.0f;
    #endif

        for( uint32_t y = 0 ; y < sy ; ++y )
        {
            for( uint32_t x = 0 ; x < sx ; ++x )
            {
                float v = (float)convOutput[f * sx * sy + y * sx + x];
                v = (v - vmin) * scale;

                v = v * 255.0f;
                v = std::min( v, 255.0f );
                v = std::max( v, 0.0f );

                DWORD col = RGB( (int)v, (int)v, (int)v );

                CRect r( x * _zoom, 
                         y * _zoom, 
                         (x + 1) * _zoom, 
                         (y + 1) * _zoom );
                r.OffsetRect( _x + f * (_zoom * sx + 3), _y );

                _dc.FillSolidRect( r, col );
            }
        }
    }

}

void BaseExample::DrawImage( CDC& _dc, const Tensor& _tensor, const TensorShape& _shape, int _x, int _y, uint32_t _zoom )
{
    if( _tensor.empty() )
        return;

    assert( _tensor.size() == _shape.Size() );

    uint32_t sx = _shape.m_SX;
    uint32_t sy = _shape.m_SY;

    for( uint32_t y = 0 ; y < sy ; ++y )
    {
        for( uint32_t x = 0 ; x < sx ; ++x )
        {
            CRect r( _x + x * _zoom, _y + y * _zoom,
                     _x + (x+1) * _zoom, _y + (y+1) * _zoom );

            Color col;

            if( _shape.m_SZ == 1 )
            {
                col.R = (float)_tensor[y * sx + x];
                col.G = col.R;
                col.B = col.R;
            }
            if( _shape.m_SZ == 3 )
            {
                col.R = (float)_tensor[0 * sx * sy + y * sx + x];
                col.G = (float)_tensor[1 * sx * sy + y * sx + x];
                col.B = (float)_tensor[2 * sx * sy + y * sx + x];
            }
            else
            {
                assert( false );
            }

            col.Saturate();

            _dc.FillSolidRect( r, col );
        }

    }
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

    m_Optimizer.LearningRate = _params.LearningRate;

    m_NeuralNet.Train( m_Optimizer, m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput,
                       numEpochs, _params.BatchSize, _params.ValidationInterval, _params.LearningRate );
}

void Example1::Draw( CDC& _dc )
{
    PlotLearningCurve( _dc, CRect( 10, 900, 800, 1200 ) );
}

//-----------------------------------------

Example2::Example2()
{
    srand( 666 );

    m_NeuralNet.AddLayer( new FullyConnected( 10 ) );
    m_NeuralNet.AddLayer( new LeakyRelu() );
    m_NeuralNet.AddLayer( new FullyConnected( 10 ) );
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
       // float y = std::sin( 0.2f * x * std::abs( std::cos( x ) ) ); //The curve we want to fit
        float y = std::sin( x );
       // float y = x < 0.0f ? 0.0f : x > 1.0f ? 0.5f : 1.0f;

        m_Input[i].push_back( x / 20.0f );
        m_ExpectedOutput[i].push_back( y );
    
        m_GroundTruthXAxis[i] = x / 20.0f;
        m_GroundTruthYAxis[i] = y;
    }

}

void Example2::GradientCheck()
{
    m_NeuralNet.GradientCheck( m_Input, m_ExpectedOutput, 500 );
}

void Example2::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_Optimizer.LearningRate = _params.LearningRate;

    m_NeuralNet.Train( m_Optimizer, m_Input, m_ExpectedOutput, m_Input, m_ExpectedOutput,
                       numEpochs, _params.BatchSize, _params.ValidationInterval );
}

void Example2::Draw( CDC& _dc )
{

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
        m_NeuralNet.AddLayer( new Convolution2D( m_NumFeatureMaps*2, m_KernelSize, m_Stride ) );
        m_NeuralNet.AddLayer( new Relu() );
        m_NeuralNet.AddLayer( new MaxPooling( 2, 2 ) );
        m_NeuralNet.AddLayer( new FullyConnected( 1000 ) );
        m_NeuralNet.AddLayer( new Relu() );
        m_NeuralNet.AddLayer( new FullyConnected( 200 ) );
        m_NeuralNet.AddLayer( new Relu() );
        m_NeuralNet.AddLayer( new FullyConnected( 10 ) );
        m_NeuralNet.AddLayer( new SoftMax() );
        m_NeuralNet.Compile( TensorShape( m_ImageRes, m_ImageRes, numChannels ) );
        m_NeuralNet.EnableClassificationAccuracyLog();

    #ifdef USE_CIFAR10_INSTEAD_OF_MNIST
        if( !LoadCifar10Dataset( "D:\\Dev\\DeepLearning Datasets\\cifar10",
                                 m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
            throw std::exception( "Can't load Cifar10 database" );
    #else
        if( !LoadMnistDataset( "D:\\Dev\\DeepLearning Datasets\\MNIST_fashion",
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

void Example3::GradientCheck()
{
    const uint32_t dataSetSize = 20;

    std::vector< Tensor > data( m_ValidationData.begin(), m_ValidationData.begin() + dataSetSize );
    std::vector< Tensor > metaData( m_ValidationMetaData.begin(), m_ValidationMetaData.begin() + dataSetSize );

    m_NeuralNet.GradientCheck( data, metaData, 50 );
}

void Example3::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_Optimizer.LearningRate = _params.LearningRate;
    m_Optimizer.WeightDecay = _params.WeightDecay;

    m_NeuralNet.Train( m_Optimizer, m_TrainingData, m_TrainingMetaData, m_ValidationData, m_ValidationMetaData,
                       numEpochs, _params.BatchSize, _params.ValidationInterval );
}

void Example3::Draw( CDC& _dc )
{
 //   Tensor out;
 //   m_NeuralNet.Evaluate( m_UserDrawnDigit, out );
 //   m_RecognizedDigit = GetMostProbableClassIndex( out );
        
    RECT r = { m_UserDrawDigitRect.left, m_UserDrawDigitRect.bottom, m_UserDrawDigitRect.right, m_UserDrawDigitRect.bottom + 30 };

    TCHAR buffer[256];
    _stprintf_s( buffer, _T("Recognized digit: %d"), m_RecognizedDigit );
    _dc.DrawText( buffer, -1, &r, DT_CENTER );
    
    DrawUserDrawnDigit( _dc );
    
    DrawConvolutionLayerFeatures( _dc, 2, 5, 5, 4 );
    DrawConvolutionLayerFeatures( _dc, 5, 5, 120, 4 );

    PlotLearningCurve( _dc, CRect( 10, 400, 800, 800 ) );

    _stprintf_s( buffer, _T( "Current accuracy: %.1f%% Best accuracy %.1f%%" ), m_NeuralNet.GetHistory().CurrentAccuracy, m_NeuralNet.GetHistory().BestAccuracy );
    _dc.DrawText( buffer, -1, &CRect( 10, 800, 800, 840 ), DT_CENTER );
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
    srand( 333 );

    const bool halfRes = true;

    m_InputShape = TensorShape( 178, 218, 3 );

    if( halfRes )
        m_InputShape = TensorShape( m_InputShape.m_SX / 2, m_InputShape.m_SY / 2, 3 );

    m_NeuralNet.AddLayer( new Convolution2D( 16, 3, 2 ) );
    m_NeuralNet.AddLayer( new Relu() );
    m_NeuralNet.AddLayer( new FullyConnected( 15 ) );
    m_NeuralNet.AddLayer( new Relu() );
    m_NeuralNet.AddLayer( new FullyConnected( 100 ) );
    m_NeuralNet.AddLayer( new Relu() );
    m_NeuralNet.AddLayer( new FullyConnected( 400 ) );
    m_NeuralNet.AddLayer( new Relu() );
    m_NeuralNet.AddLayer( new FullyConnected( m_InputShape.Size() ) );
    m_NeuralNet.AddLayer( new Sigmoid() );
    m_NeuralNet.Compile( m_InputShape );

    if( !LoadCelebADataset( "D:\\Dev\\DeepLearning Datasets\\CelebA", halfRes, 2.0f, 0.02f,
                            m_TrainingData, m_ValidationData, m_TrainingMetaData, m_ValidationMetaData ) )
        throw std::exception("Can't load celebA database");
}

void Example4::Train( const HyperParameters& _params )
{
    //Hyper parameters
    const int numEpochs = 10000000;
    const float errorTarget = 0.0000001f;

    m_Optimizer.LearningRate = _params.LearningRate;

    m_NeuralNet.Train( m_Optimizer, m_TrainingData, m_TrainingData, m_ValidationData, m_ValidationData,
                       numEpochs, _params.BatchSize, _params.ValidationInterval );
}

void Example4::Draw( CDC& _dc )
{
    PlotLearningCurve( _dc, CRect( 10, 400, 800, 800 ) );
    DrawConvolutionLayerFeatures( _dc, 1, 5, 5, 3 );

    const Layer* outputLayer = m_NeuralNet.DbgGetLayer( m_NeuralNet.DbgGetLayerCount() - 1 );

    DrawImage( _dc, outputLayer->GetOutput(), m_InputShape, 1000, 300, 3 );
}
