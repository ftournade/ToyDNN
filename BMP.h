#pragma once

#include <stdint.h>

bool WriteBMP( const char* fname, const uint8_t* buf, uint32_t width,	uint32_t height );