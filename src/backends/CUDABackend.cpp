#ifdef USE_CUDA
#include "backends/CUDABackend.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

// Launcher declarations (implemented in filters.cu)
extern "C" {
    void launch_grayscale(const uint8_t*, uint8_t*, int, int, cudaStream_t);
    void launch_gaussian_blur(const uint8_t*, uint8_t*, int, int, cudaStream_t);
    void launch_sobel_edge(const uint8_t*, uint8_t*, int, int, cudaStream_t);
    void launch_bilateral_filter(const uint8_t*, uint8_t*, int, int, cudaStream_t);
    void launch_histogram_eq(const uint8_t*, uint8_t*, int, int,
        const uint8_t*, const uint8_t*, const uint8_t*, cudaStream_t);
    void launch_kuwahara(const uint8_t*, uint8_t*, int, int, cudaStream_t);
}

CUDABackend::CUDABackend() {
    int count = 0;
    available_ = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

std::string CUDABackend::name()      const { return "cuda"; }
bool        CUDABackend::available() const { return available_; }

static std::vector<uint8_t> build_lut_cpu(const uint8_t* in, int w, int h, int ch_off) {
    int hist[256] = {};
    for (int i = 0;i < w * h;i++) hist[in[i * 4 + ch_off]]++;
    int cdf[256] = {}; cdf[0] = hist[0];
    for (int v = 1;v < 256;v++) cdf[v] = cdf[v - 1] + hist[v];
    int cdf_min = 0; for (int v = 0;v < 256;v++) if (cdf[v] > 0) { cdf_min = cdf[v];break; }
    int denom = w * h - cdf_min;
    std::vector<uint8_t> lut(256);
    for (int v = 0;v < 256;v++)
        lut[v] = (denom == 0) ? v : (uint8_t)std::clamp(
            (int)std::round((float)(cdf[v] - cdf_min) / (float)denom * 255.f), 0, 255);
    return lut;
}

BenchmarkResult CUDABackend::run(const IFilter& filter, const Image& input) const {
    if (!available_) throw std::runtime_error("No CUDA device");

    std::string fn = filter.name();
    if (fn != "grayscale" && fn != "gaussian_blur" && fn != "sobel_edge" &&
        fn != "bilateral_filter" && fn != "histogram_eq" && fn != "kuwahara")
        throw std::runtime_error("No CUDA kernel for filter: " + fn);

    int w = input.width(), h = input.height();
    int grid_x = (w + 15) / 16, grid_y = (h + 15) / 16;
    std::cout << "[cuda] grid=" << grid_x << "x" << grid_y
        << " (" << grid_x * grid_y * 256 << " GPU threads)\n";
    size_t bytes = input.size();

    uint8_t* d_in = nullptr, * d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    // Histogram eq LUTs (computed on host, uploaded)
    uint8_t* d_lut_r = nullptr, * d_lut_g = nullptr, * d_lut_b = nullptr;
    if (filter.name() == "histogram_eq") {
        auto lr = build_lut_cpu(input.data(), w, h, 0);
        auto lg = build_lut_cpu(input.data(), w, h, 1);
        auto lb = build_lut_cpu(input.data(), w, h, 2);
        cudaMalloc(&d_lut_r, 256); cudaMemcpy(d_lut_r, lr.data(), 256, cudaMemcpyHostToDevice);
        cudaMalloc(&d_lut_g, 256); cudaMemcpy(d_lut_g, lg.data(), 256, cudaMemcpyHostToDevice);
        cudaMalloc(&d_lut_b, 256); cudaMemcpy(d_lut_b, lb.data(), 256, cudaMemcpyHostToDevice);
    }

    cudaEventRecord(ev0, stream);

    if (fn == "grayscale")       launch_grayscale(d_in, d_out, w, h, stream);
    else if (fn == "gaussian_blur")   launch_gaussian_blur(d_in, d_out, w, h, stream);
    else if (fn == "sobel_edge")      launch_sobel_edge(d_in, d_out, w, h, stream);
    else if (fn == "bilateral_filter")launch_bilateral_filter(d_in, d_out, w, h, stream);
    else if (fn == "histogram_eq")    launch_histogram_eq(d_in, d_out, w, h,
        d_lut_r, d_lut_g, d_lut_b, stream);
    else if (fn == "kuwahara")        launch_kuwahara(d_in, d_out, w, h, stream);

    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, ev0, ev1);

    BenchmarkResult result;
    result.backend_name = name();
    result.filter_name = filter.name();
    result.elapsed_ms = elapsed_ms;
    result.output = Image(w, h, 4);
    cudaMemcpy(result.output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out);
    if (d_lut_r) { cudaFree(d_lut_r);cudaFree(d_lut_g);cudaFree(d_lut_b); }
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    return result;
}
#endif // USE_CUDA
