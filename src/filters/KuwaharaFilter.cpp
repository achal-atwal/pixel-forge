#include "filters/KuwaharaFilter.hpp"
#include "core/parallel.hpp"
#include <cmath>
#include <algorithm>

// For each output pixel, divide surrounding area into 4 overlapping quadrants
// (each (r+1)x(r+1), sharing edges at the center pixel).  Output = mean of
// the quadrant with the lowest luminance variance — preserves edges while
// smoothing uniform regions.
static void kuwahara_rows(const uint8_t* in, uint8_t* out, int w, int h,
    int row_start, int row_end, int r) {
    // Quadrant offsets: [dx_lo, dx_hi, dy_lo, dy_hi]
    const int qx0[4] = { -r, 0, -r, 0 };
    const int qx1[4] = { 0, r,  0, r };
    const int qy0[4] = { -r,-r,  0, 0 };
    const int qy1[4] = { 0, 0,  r, r };

    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < w; x++) {
            float best_var = 1e30f;
            float best_r = 0, best_g = 0, best_b = 0;

            for (int q = 0; q < 4; q++) {
                float sr = 0, sg = 0, sb = 0, sl = 0, sl2 = 0;
                int count = 0;
                for (int dy = qy0[q]; dy <= qy1[q]; dy++) {
                    int ny = std::clamp(y + dy, 0, h - 1);
                    for (int dx = qx0[q]; dx <= qx1[q]; dx++) {
                        int nx = std::clamp(x + dx, 0, w - 1);
                        const uint8_t* p = in + (ny * w + nx) * 4;
                        float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
                        sr += p[0]; sg += p[1]; sb += p[2];
                        sl += luma; sl2 += luma * luma;
                        count++;
                    }
                }
                float inv = 1.f / count;
                float mean_l = sl * inv;
                float var = sl2 * inv - mean_l * mean_l;
                if (var < best_var) {
                    best_var = var;
                    best_r = sr * inv;
                    best_g = sg * inv;
                    best_b = sb * inv;
                }
            }

            const uint8_t* cp = in + (y * w + x) * 4;
            uint8_t* qp = out + (y * w + x) * 4;
            qp[0] = (uint8_t)std::clamp((int)best_r, 0, 255);
            qp[1] = (uint8_t)std::clamp((int)best_g, 0, 255);
            qp[2] = (uint8_t)std::clamp((int)best_b, 0, 255);
            qp[3] = cp[3];
        }
    }
}

std::string KuwaharaFilter::name() const { return "kuwahara"; }

void KuwaharaFilter::apply(const uint8_t* in, uint8_t* out, int w, int h, int) const {
    kuwahara_rows(in, out, w, h, 0, h, kRadius);
}

void KuwaharaFilter::apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int) const {
    parallel_for(h, [&](int y) {
        kuwahara_rows(in, out, w, h, y, y + 1, kRadius);
        });
}
