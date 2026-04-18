#include "filters/HistogramEqualization.hpp"
#include "core/parallel.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <cmath>

// Build equalization LUT for one channel
static void build_lut(const uint8_t* in, int w, int h, int ch, int c_offset,
    uint8_t lut[256]) {
    int hist[256] = {};
    int total = w * h;
    for (int i = 0; i < total; i++)
        hist[in[i * ch + c_offset]]++;

    // CDF
    int cdf[256] = {};
    cdf[0] = hist[0];
    for (int v = 1;v < 256;v++) cdf[v] = cdf[v - 1] + hist[v];

    // Find CDF min (first non-zero)
    int cdf_min = 0;
    for (int v = 0;v < 256;v++) { if (cdf[v] > 0) { cdf_min = cdf[v]; break; } }

    int denom = total - cdf_min;
    for (int v = 0;v < 256;v++) {
        if (denom == 0) { lut[v] = v; continue; }
        lut[v] = static_cast<uint8_t>(
            std::clamp((int)std::round((float)(cdf[v] - cdf_min) / (float)denom * 255.f), 0, 255));
    }
}

std::string HistogramEqualization::name() const { return "histogram_eq"; }

void HistogramEqualization::apply(const uint8_t* in, uint8_t* out, int w, int h, int ch) const {
    uint8_t lut_r[256], lut_g[256], lut_b[256];
    build_lut(in, w, h, ch, 0, lut_r);
    build_lut(in, w, h, ch, 1, lut_g);
    build_lut(in, w, h, ch, 2, lut_b);
    for (int i = 0; i < w * h; i++) {
        out[i * ch + 0] = lut_r[in[i * ch + 0]];
        out[i * ch + 1] = lut_g[in[i * ch + 1]];
        out[i * ch + 2] = lut_b[in[i * ch + 2]];
        if (ch == 4) out[i * ch + 3] = in[i * ch + 3];
    }
}

void HistogramEqualization::apply_parallel(const uint8_t* in, uint8_t* out,
    int w, int h, int ch) const {
    uint8_t lut_r[256], lut_g[256], lut_b[256];
    build_lut(in, w, h, ch, 0, lut_r);
    build_lut(in, w, h, ch, 1, lut_g);
    build_lut(in, w, h, ch, 2, lut_b);
    parallel_for(w * h, [&](int i) {
        out[i * ch + 0] = lut_r[in[i * ch + 0]];
        out[i * ch + 1] = lut_g[in[i * ch + 1]];
        out[i * ch + 2] = lut_b[in[i * ch + 2]];
        if (ch == 4) out[i * ch + 3] = in[i * ch + 3];
        });
}
