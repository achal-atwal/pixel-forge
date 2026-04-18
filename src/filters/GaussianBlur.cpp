#include "filters/GaussianBlur.hpp"
#include "core/parallel.hpp"
#include <algorithm>
#include <vector>
#include <cstdint>

constexpr float GaussianBlur::kKernel[7];

static void blur_pass_h(const uint8_t* in, uint8_t* out, int w, int h,
    const float k[7], int row_start, int row_end) {
    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < w; x++) {
            float r = 0, g = 0, b = 0;
            for (int d = -3; d <= 3; d++) {
                int xi = std::clamp(x + d, 0, w - 1);
                const uint8_t* p = in + (y * w + xi) * 4;
                r += k[d + 3] * p[0]; g += k[d + 3] * p[1]; b += k[d + 3] * p[2];
            }
            uint8_t* q = out + (y * w + x) * 4;
            q[0] = static_cast<uint8_t>(r); q[1] = static_cast<uint8_t>(g);
            q[2] = static_cast<uint8_t>(b); q[3] = in[(y * w + x) * 4 + 3];
        }
    }
}

static void blur_pass_v(const uint8_t* in, uint8_t* out, int w, int h,
    const float k[7], int row_start, int row_end) {
    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < w; x++) {
            float r = 0, g = 0, b = 0;
            for (int d = -3; d <= 3; d++) {
                int yi = std::clamp(y + d, 0, h - 1);
                const uint8_t* p = in + (yi * w + x) * 4;
                r += k[d + 3] * p[0]; g += k[d + 3] * p[1]; b += k[d + 3] * p[2];
            }
            uint8_t* q = out + (y * w + x) * 4;
            q[0] = static_cast<uint8_t>(r); q[1] = static_cast<uint8_t>(g);
            q[2] = static_cast<uint8_t>(b); q[3] = in[(y * w + x) * 4 + 3];
        }
    }
}

std::string GaussianBlur::name() const { return "gaussian_blur"; }

void GaussianBlur::apply(const uint8_t* in, uint8_t* out, int w, int h, int /*ch*/) const {
    std::vector<uint8_t> tmp(w * h * 4);
    blur_pass_h(in, tmp.data(), w, h, kKernel, 0, h);
    blur_pass_v(tmp.data(), out, w, h, kKernel, 0, h);
}

void GaussianBlur::apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int /*ch*/) const {
    std::vector<uint8_t> tmp(w * h * 4);
    parallel_for(h, [&](int y) {
        blur_pass_h(in, tmp.data(), w, h, kKernel, y, y + 1);
        });
    parallel_for(h, [&](int y) {
        blur_pass_v(tmp.data(), out, w, h, kKernel, y, y + 1);
        });
}
