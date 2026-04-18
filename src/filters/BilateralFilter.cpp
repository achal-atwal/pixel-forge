#include "filters/BilateralFilter.hpp"
#include "core/parallel.hpp"
#include <cmath>
#include <algorithm>

static void bilateral_rows(const uint8_t* in, uint8_t* out, int w, int h,
    int row_start, int row_end,
    float inv2ss2, float inv2sr2, int radius) {
    for (int y = row_start; y < row_end; y++) {
        for (int x = 0; x < w; x++) {
            const uint8_t* cp = in + (y * w + x) * 4;
            float sr = 0, sg = 0, sb = 0, wsum = 0;
            for (int dy = -radius; dy <= radius; dy++) {
                int ny = std::clamp(y + dy, 0, h - 1);
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = std::clamp(x + dx, 0, w - 1);
                    const uint8_t* np = in + (ny * w + nx) * 4;
                    float ds = (float)(dx * dx + dy * dy);
                    float dr0 = (float)cp[0] - (float)np[0];
                    float dr1 = (float)cp[1] - (float)np[1];
                    float dr2 = (float)cp[2] - (float)np[2];
                    float dr = dr0 * dr0 + dr1 * dr1 + dr2 * dr2;
                    float w_val = std::exp(-ds * inv2ss2 - dr * inv2sr2);
                    sr += w_val * np[0]; sg += w_val * np[1];
                    sb += w_val * np[2]; wsum += w_val;
                }
            }
            uint8_t* q = out + (y * w + x) * 4;
            q[0] = static_cast<uint8_t>(sr / wsum);
            q[1] = static_cast<uint8_t>(sg / wsum);
            q[2] = static_cast<uint8_t>(sb / wsum);
            q[3] = cp[3];
        }
    }
}

std::string BilateralFilter::name() const { return "bilateral_filter"; }

void BilateralFilter::apply(const uint8_t* in, uint8_t* out, int w, int h, int) const {
    float inv2ss2 = 1.f / (2.f * kSigmaS * kSigmaS);
    float inv2sr2 = 1.f / (2.f * kSigmaR * kSigmaR);
    bilateral_rows(in, out, w, h, 0, h, inv2ss2, inv2sr2, kRadius);
}

void BilateralFilter::apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int) const {
    float inv2ss2 = 1.f / (2.f * kSigmaS * kSigmaS);
    float inv2sr2 = 1.f / (2.f * kSigmaR * kSigmaR);
    parallel_for(h, [&](int y) {
        bilateral_rows(in, out, w, h, y, y + 1, inv2ss2, inv2sr2, kRadius);
        });
}
