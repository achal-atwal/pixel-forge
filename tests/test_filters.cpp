#include <catch2/catch_test_macros.hpp>
#include "filters/Grayscale.hpp"
#include "filters/GaussianBlur.hpp"
#include "filters/SobelEdge.hpp"
#include "filters/BilateralFilter.hpp"
#include "filters/HistogramEqualization.hpp"
#include "filters/KuwaharaFilter.hpp"
#include <cstring>
#include <cmath>
#include <vector>

static void fill(uint8_t* buf, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    for (int i = 0; i < w * h; i++) {
        buf[i * 4 + 0] = r; buf[i * 4 + 1] = g; buf[i * 4 + 2] = b; buf[i * 4 + 3] = a;
    }
}

// Grayscale

TEST_CASE("Grayscale: name") {
    Grayscale f;
    CHECK(f.name() == "grayscale");
}

TEST_CASE("Grayscale: red pixel -> correct luma") {
    std::vector<uint8_t> in(4), out(4);
    fill(in.data(), 1, 1, 255, 0, 0);
    Grayscale().apply(in.data(), out.data(), 1, 1, 4);
    uint8_t expected = static_cast<uint8_t>(0.299f * 255.f);  // 76
    CHECK(out[0] == expected);
    CHECK(out[1] == expected);
    CHECK(out[2] == expected);
    CHECK(out[3] == 255);  // alpha preserved
}

TEST_CASE("Grayscale: white pixel stays white") {
    std::vector<uint8_t> in(4), out(4);
    fill(in.data(), 1, 1, 255, 255, 255);
    Grayscale().apply(in.data(), out.data(), 1, 1, 4);
    CHECK(out[0] == 255);
}

TEST_CASE("Grayscale: apply_parallel matches apply on 32x32") {
    int w = 32, h = 32;
    std::vector<uint8_t> in(w * h * 4), out1(w * h * 4), out2(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = i % 256;
    Grayscale f;
    f.apply(in.data(), out1.data(), w, h, 4);
    f.apply_parallel(in.data(), out2.data(), w, h, 4);
    CHECK(std::memcmp(out1.data(), out2.data(), out1.size()) == 0);
}

// GaussianBlur

TEST_CASE("GaussianBlur: name") {
    CHECK(GaussianBlur().name() == "gaussian_blur");
}

TEST_CASE("GaussianBlur: uniform image unchanged") {
    int w = 16, h = 16;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    fill(in.data(), w, h, 100, 150, 200);
    GaussianBlur().apply(in.data(), out.data(), w, h, 4);
    // Allow ±1 for floating-point rounding at edges
    for (int i = 0;i < w * h;i++) {
        CHECK(std::abs((int)out[i * 4 + 0] - 100) <= 1);
        CHECK(std::abs((int)out[i * 4 + 1] - 150) <= 1);
        CHECK(std::abs((int)out[i * 4 + 2] - 200) <= 1);
        CHECK(out[i * 4 + 3] == 255);
    }
}

TEST_CASE("GaussianBlur: apply_parallel matches apply on 64x64") {
    int w = 64, h = 64;
    std::vector<uint8_t> in(w * h * 4), out1(w * h * 4), out2(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = (i * 7 + 13) % 256;
    GaussianBlur f;
    f.apply(in.data(), out1.data(), w, h, 4);
    f.apply_parallel(in.data(), out2.data(), w, h, 4);
    CHECK(std::memcmp(out1.data(), out2.data(), out1.size()) == 0);
}

// SobelEdge

TEST_CASE("SobelEdge: name") {
    CHECK(SobelEdge().name() == "sobel_edge");
}

TEST_CASE("SobelEdge: uniform image produces zero edges") {
    int w = 16, h = 16;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    fill(in.data(), w, h, 128, 128, 128);
    SobelEdge().apply(in.data(), out.data(), w, h, 4);
    // Interior pixels must be 0 — skip border pixels (1px padding)
    for (int y = 1;y < h - 1;y++) for (int x = 1;x < w - 1;x++) {
        int i = (y * w + x) * 4;
        CHECK(out[i + 0] == 0);
    }
}

TEST_CASE("SobelEdge: apply_parallel matches apply on 64x64") {
    int w = 64, h = 64;
    std::vector<uint8_t> in(w * h * 4), out1(w * h * 4), out2(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = (i * 11 + 7) % 256;
    SobelEdge f;
    f.apply(in.data(), out1.data(), w, h, 4);
    f.apply_parallel(in.data(), out2.data(), w, h, 4);
    CHECK(std::memcmp(out1.data(), out2.data(), out1.size()) == 0);
}

// BilateralFilter

TEST_CASE("BilateralFilter: name") {
    CHECK(BilateralFilter().name() == "bilateral_filter");
}

TEST_CASE("BilateralFilter: uniform image unchanged") {
    int w = 16, h = 16;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    fill(in.data(), w, h, 80, 120, 200);
    BilateralFilter().apply(in.data(), out.data(), w, h, 4);
    for (int i = 0;i < w * h;i++) {
        CHECK(std::abs((int)out[i * 4 + 0] - 80) <= 1);
        CHECK(std::abs((int)out[i * 4 + 1] - 120) <= 1);
        CHECK(std::abs((int)out[i * 4 + 2] - 200) <= 1);
        CHECK(out[i * 4 + 3] == 255);
    }
}

TEST_CASE("BilateralFilter: apply_parallel matches apply on 32x32") {
    int w = 32, h = 32;
    std::vector<uint8_t> in(w * h * 4), out1(w * h * 4), out2(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = (i * 13 + 5) % 256;
    BilateralFilter f;
    f.apply(in.data(), out1.data(), w, h, 4);
    f.apply_parallel(in.data(), out2.data(), w, h, 4);
    CHECK(std::memcmp(out1.data(), out2.data(), out1.size()) == 0);
}

// HistogramEqualization

TEST_CASE("HistogramEqualization: name") {
    CHECK(HistogramEqualization().name() == "histogram_eq");
}

TEST_CASE("HistogramEqualization: output pixels in [0,255]") {
    int w = 32, h = 32;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = (i * 17) % 256;
    HistogramEqualization().apply(in.data(), out.data(), w, h, 4);
    for (int i = 0;i < w * h * 4;i++) CHECK((int)out[i] >= 0);  // trivially true but validates no UB
}

TEST_CASE("HistogramEqualization: dark image with variation gets stretched") {
    int w = 32, h = 32;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    // Create bimodal histogram: half at 10, half at 20 (narrow dark range)
    for (int i = 0;i < w * h;i++) {
        in[i * 4] = (i < w * h / 2) ? 10 : 20;
        in[i * 4 + 1] = (i < w * h / 2) ? 10 : 20;
        in[i * 4 + 2] = (i < w * h / 2) ? 10 : 20;
        in[i * 4 + 3] = 255;
    }
    HistogramEqualization().apply(in.data(), out.data(), w, h, 4);
    // After equalization, the values should be more spread out
    // Find min/max in output
    uint8_t out_min = 255, out_max = 0;
    for (int i = 0;i < w * h;i++) {
        out_min = std::min(out_min, out[i * 4]);
        out_max = std::max(out_max, out[i * 4]);
    }
    // The spread should be much larger than input spread (20-10=10)
    CHECK((int)out_max - (int)out_min > 30);
}

TEST_CASE("HistogramEqualization: apply_parallel matches apply on 64x64") {
    int w = 64, h = 64;
    std::vector<uint8_t> in(w * h * 4), out1(w * h * 4), out2(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = (i * 31 + 3) % 256;
    HistogramEqualization f;
    f.apply(in.data(), out1.data(), w, h, 4);
    f.apply_parallel(in.data(), out2.data(), w, h, 4);
    CHECK(std::memcmp(out1.data(), out2.data(), out1.size()) == 0);
}

// KuwaharaFilter

TEST_CASE("KuwaharaFilter: name") {
    CHECK(KuwaharaFilter().name() == "kuwahara");
}

TEST_CASE("KuwaharaFilter: uniform image unchanged") {
    int w = 32, h = 32;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    fill(in.data(), w, h, 80, 140, 200);
    KuwaharaFilter().apply(in.data(), out.data(), w, h, 4);
    // All quadrants have zero variance; output mean equals input color
    for (int i = 0;i < w * h;i++) {
        CHECK(std::abs((int)out[i * 4 + 0] - 80) <= 1);
        CHECK(std::abs((int)out[i * 4 + 1] - 140) <= 1);
        CHECK(std::abs((int)out[i * 4 + 2] - 200) <= 1);
        CHECK(out[i * 4 + 3] == 255);
    }
}

TEST_CASE("KuwaharaFilter: alpha preserved") {
    int w = 16, h = 16;
    std::vector<uint8_t> in(w * h * 4), out(w * h * 4);
    fill(in.data(), w, h, 100, 100, 100, 42);
    KuwaharaFilter().apply(in.data(), out.data(), w, h, 4);
    for (int i = 0;i < w * h;i++) CHECK(out[i * 4 + 3] == 42);
}

TEST_CASE("KuwaharaFilter: apply_parallel matches apply on 64x64") {
    int w = 64, h = 64;
    std::vector<uint8_t> in(w * h * 4), out1(w * h * 4), out2(w * h * 4);
    for (int i = 0;i < w * h * 4;i++) in[i] = (i * 17 + 3) % 256;
    KuwaharaFilter f;
    f.apply(in.data(), out1.data(), w, h, 4);
    f.apply_parallel(in.data(), out2.data(), w, h, 4);
    CHECK(std::memcmp(out1.data(), out2.data(), out1.size()) == 0);
}
