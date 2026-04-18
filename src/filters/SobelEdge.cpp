#include "filters/SobelEdge.hpp"
#include "core/parallel.hpp"
#include <cmath>
#include <algorithm>

// Returns luma of pixel at (x,y), clamping to border
static inline float luma_at(const uint8_t* in, int w, int h, int x, int y) {
    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    const uint8_t* p = in + (y * w + x) * 4;
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
}

static void sobel_pixel(const uint8_t* in, uint8_t* out, int w, int h,
    int row_start, int row_end) {
    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < w; x++) {
            float gx = -luma_at(in, w, h, x - 1, y - 1) + luma_at(in, w, h, x + 1, y - 1)
                - 2 * luma_at(in, w, h, x - 1, y) + 2 * luma_at(in, w, h, x + 1, y)
                - luma_at(in, w, h, x - 1, y + 1) + luma_at(in, w, h, x + 1, y + 1);
            float gy = -luma_at(in, w, h, x - 1, y - 1) - 2 * luma_at(in, w, h, x, y - 1)
                - luma_at(in, w, h, x + 1, y - 1)
                + luma_at(in, w, h, x - 1, y + 1) + 2 * luma_at(in, w, h, x, y + 1)
                + luma_at(in, w, h, x + 1, y + 1);
            uint8_t mag = static_cast<uint8_t>(std::min(std::sqrt(gx * gx + gy * gy), 255.f));
            uint8_t* q = out + (y * w + x) * 4;
            q[0] = mag; q[1] = mag; q[2] = mag; q[3] = in[(y * w + x) * 4 + 3];
        }
    }
}

std::string SobelEdge::name() const { return "sobel_edge"; }

void SobelEdge::apply(const uint8_t* in, uint8_t* out, int w, int h, int) const {
    sobel_pixel(in, out, w, h, 0, h);
}

void SobelEdge::apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int) const {
    parallel_for(h, [&](int y) {
        sobel_pixel(in, out, w, h, y, y + 1);
        });
}
