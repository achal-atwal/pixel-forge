#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// Grayscale
__global__ void k_grayscale(const uint8_t* in, uint8_t* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = (y * w + x) * 4;
    uint8_t luma = (uint8_t)(0.299f * in[i] + 0.587f * in[i + 1] + 0.114f * in[i + 2]);
    out[i] = luma; out[i + 1] = luma; out[i + 2] = luma; out[i + 3] = in[i + 3];
}

// Gaussian Blur (2D 5x5 kernel)
__global__ void k_gaussian_blur(const uint8_t* in, uint8_t* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float k[7] = { 0.0702f,0.1311f,0.1907f,0.2161f,0.1907f,0.1311f,0.0702f };
    float r = 0, g = 0, b = 0;
    for (int dy = -3;dy <= 3;dy++) for (int dx = -3;dx <= 3;dx++) {
        int nx = max(0, min(w - 1, x + dx)), ny = max(0, min(h - 1, y + dy));
        const uint8_t* p = in + (ny * w + nx) * 4;
        float kw = k[dx + 3] * k[dy + 3];
        r += kw * p[0]; g += kw * p[1]; b += kw * p[2];
    }
    int i = (y * w + x) * 4;
    out[i] = (uint8_t)r; out[i + 1] = (uint8_t)g; out[i + 2] = (uint8_t)b; out[i + 3] = in[i + 3];
}

// Sobel Edge
__global__ void k_sobel_edge(const uint8_t* in, uint8_t* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    auto luma = [&](int px, int py)->float {
        px = max(0, min(w - 1, px)); py = max(0, min(h - 1, py));
        const uint8_t* p = in + (py * w + px) * 4;
        return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
        };
    float gx = -luma(x - 1, y - 1) + luma(x + 1, y - 1) - 2 * luma(x - 1, y) + 2 * luma(x + 1, y)
        - luma(x - 1, y + 1) + luma(x + 1, y + 1);
    float gy = -luma(x - 1, y - 1) - 2 * luma(x, y - 1) - luma(x + 1, y - 1)
        + luma(x - 1, y + 1) + 2 * luma(x, y + 1) + luma(x + 1, y + 1);
    uint8_t mag = (uint8_t)fminf(sqrtf(gx * gx + gy * gy), 255.f);
    int i = (y * w + x) * 4;
    out[i] = mag; out[i + 1] = mag; out[i + 2] = mag; out[i + 3] = in[i + 3];
}

// Bilateral Filter
__global__ void k_bilateral_filter(const uint8_t* in, uint8_t* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float inv2ss2 = 1.f / 50.f, inv2sr2 = 1.f / 3200.f;
    const uint8_t* cp = in + (y * w + x) * 4;
    float sr = 0, sg = 0, sb = 0, wsum = 0;
    for (int dy = -7;dy <= 7;dy++) for (int dx = -7;dx <= 7;dx++) {
        int nx = max(0, min(w - 1, x + dx)), ny = max(0, min(h - 1, y + dy));
        const uint8_t* np = in + (ny * w + nx) * 4;
        float ds = (float)(dx * dx + dy * dy);
        float dr0 = (float)cp[0] - (float)np[0],
            dr1 = (float)cp[1] - (float)np[1],
            dr2 = (float)cp[2] - (float)np[2];
        float wv = expf(-ds * inv2ss2 - (dr0 * dr0 + dr1 * dr1 + dr2 * dr2) * inv2sr2);
        sr += wv * np[0]; sg += wv * np[1]; sb += wv * np[2]; wsum += wv;
    }
    int i = (y * w + x) * 4;
    out[i] = (uint8_t)(sr / wsum); out[i + 1] = (uint8_t)(sg / wsum);
    out[i + 2] = (uint8_t)(sb / wsum); out[i + 3] = cp[3];
}

// Histogram Equalization
__global__ void k_histogram_eq(const uint8_t* in, uint8_t* out, int w, int h,
    const uint8_t* lut_r, const uint8_t* lut_g,
    const uint8_t* lut_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = (y * w + x) * 4;
    out[i] = lut_r[in[i]]; out[i + 1] = lut_g[in[i + 1]];
    out[i + 2] = lut_b[in[i + 2]]; out[i + 3] = in[i + 3];
}

// Kuwahara Filter
__global__ void k_kuwahara(const uint8_t* in, uint8_t* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int qx0[4] = { -5, 0,-5, 0 };
    const int qx1[4] = { 0, 5, 0, 5 };
    const int qy0[4] = { -5,-5, 0, 0 };
    const int qy1[4] = { 0, 0, 5, 5 };

    float best_var = 1e30f, best_r = 0, best_g = 0, best_b = 0;
    for (int q = 0;q < 4;q++) {
        float sr = 0, sg = 0, sb = 0, sl = 0, sl2 = 0; int count = 0;
        for (int dy = qy0[q];dy <= qy1[q];dy++) {
            int ny = max(0, min(h - 1, y + dy));
            for (int dx = qx0[q];dx <= qx1[q];dx++) {
                int nx = max(0, min(w - 1, x + dx));
                const uint8_t* p = in + (ny * w + nx) * 4;
                float luma = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
                sr += p[0]; sg += p[1]; sb += p[2];
                sl += luma; sl2 += luma * luma; count++;
            }
        }
        float inv = 1.f / count, mean_l = sl * inv;
        float var = sl2 * inv - mean_l * mean_l;
        if (var < best_var) { best_var = var; best_r = sr * inv; best_g = sg * inv; best_b = sb * inv; }
    }
    int i = (y * w + x) * 4;
    out[i] = (uint8_t)fminf(fmaxf(best_r, 0.f), 255.f);
    out[i + 1] = (uint8_t)fminf(fmaxf(best_g, 0.f), 255.f);
    out[i + 2] = (uint8_t)fminf(fmaxf(best_b, 0.f), 255.f);
    out[i + 3] = in[i + 3];
}

// Launcher functions (called from CUDABackend.cpp)
extern "C" {

    void launch_grayscale(const uint8_t* in, uint8_t* out, int w, int h,
        cudaStream_t stream) {
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        k_grayscale << <grid, block, 0, stream >> > (in, out, w, h);
    }
    void launch_gaussian_blur(const uint8_t* in, uint8_t* out, int w, int h,
        cudaStream_t stream) {
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        k_gaussian_blur << <grid, block, 0, stream >> > (in, out, w, h);
    }
    void launch_sobel_edge(const uint8_t* in, uint8_t* out, int w, int h,
        cudaStream_t stream) {
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        k_sobel_edge << <grid, block, 0, stream >> > (in, out, w, h);
    }
    void launch_bilateral_filter(const uint8_t* in, uint8_t* out, int w, int h,
        cudaStream_t stream) {
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        k_bilateral_filter << <grid, block, 0, stream >> > (in, out, w, h);
    }
    void launch_histogram_eq(const uint8_t* in, uint8_t* out, int w, int h,
        const uint8_t* lut_r, const uint8_t* lut_g,
        const uint8_t* lut_b, cudaStream_t stream) {
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        k_histogram_eq << <grid, block, 0, stream >> > (in, out, w, h, lut_r, lut_g, lut_b);
    }

    void launch_kuwahara(const uint8_t* in, uint8_t* out, int w, int h,
        cudaStream_t stream) {
        dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
        k_kuwahara << <grid, block, 0, stream >> > (in, out, w, h);
    }

} // extern "C"
