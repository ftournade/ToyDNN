#include "pch.h"
#include "Layers/FullyConnectedLayer.h"
#include "Layers/Convolution2DLayer.h"
#include "Layers/MaxPoolingLayer.h"
#include "Layers/Activation/ActivationLayers.h"
#include "Layers/PaddingLayer.h"

namespace ToyDNN
{
	Layer* CreateLayer( LayerType _type )
	{
		switch( _type )
		{
			case LayerType::FullyConnected: return new FullyConnected;
			case LayerType::Convolution2D: return new Convolution2D;
			case LayerType::ConvolutionTranspose2D: return new ConvolutionTranspose2D;
			case LayerType::MaxPooling: return new MaxPooling;
				 
			case LayerType::Relu: return new Relu;
			case LayerType::LeakyRelu: return new LeakyRelu;
			case LayerType::Sigmoid: return new Sigmoid;
			case LayerType::Tanh: return new Tanh;
//			case LayerType::SoftMax: return new SoftMax;
			case LayerType::Padding: return new PaddingLayer;
			default: return nullptr;
		}
	}

}
