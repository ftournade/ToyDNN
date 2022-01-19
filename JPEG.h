#pragma once

#include <vector>

bool LoadJpeg( const char* fname, std::vector< uint8_t >& _pixels, uint32_t& _width, uint32_t& _height );
