#pragma once

#include <vector>

typedef std::vector< float > Tensor;

bool WriteBMP( const char* _filename, bool _grayscale, const Tensor& _pixels, int _width, int _height ); //TODO put that somewhere else
