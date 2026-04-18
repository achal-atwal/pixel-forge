#include "filters/Grayscale.hpp"
#include "core/parallel.hpp"

std::string Grayscale::name() const { return "grayscale"; }

void Grayscale::apply(const uint8_t* in, uint8_t* out, int w, int h, int ch) const {
    for (int i = 0; i < w * h; i++) {
        const uint8_t* p = in + i * ch;
        uint8_t* q = out + i * ch;
        uint8_t luma = static_cast<uint8_t>(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
        q[0] = luma; q[1] = luma; q[2] = luma;
        if (ch == 4) q[3] = p[3];
    }
}

void Grayscale::apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int ch) const {
    parallel_for(w * h, [&](int i) {
        const uint8_t* p = in + i * ch;
        uint8_t* q = out + i * ch;
        uint8_t luma = static_cast<uint8_t>(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
        q[0] = luma; q[1] = luma; q[2] = luma;
        if (ch == 4) q[3] = p[3];
        });
}
