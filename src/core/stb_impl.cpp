// Single translation unit that instantiates stb_image implementations.
// No other file should define STB_IMAGE_IMPLEMENTATION or STB_IMAGE_WRITE_IMPLEMENTATION.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
