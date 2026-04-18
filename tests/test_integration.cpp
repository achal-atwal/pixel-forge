#include <catch2/catch_test_macros.hpp>
#include "core/Image.hpp"
#include "filters/Grayscale.hpp"
#include "filters/GaussianBlur.hpp"
#include "filters/SobelEdge.hpp"
#include "filters/BilateralFilter.hpp"
#include "filters/HistogramEqualization.hpp"
#include "backends/CPUSingleBackend.hpp"
#include "backends/CPUMultiBackend.hpp"
#include "core/Benchmark.hpp"
#include <cstring>
#include <sstream>

// Helper: create a synthetic 64x64 gradient image
static Image make_gradient(int w = 64, int h = 64) {
    Image img(w, h, 4);
    for (int y = 0;y < h;y++) for (int x = 0;x < w;x++) {
        int i = (y * w + x) * 4;
        img.data()[i + 0] = (uint8_t)(x * 255 / w);
        img.data()[i + 1] = (uint8_t)(y * 255 / h);
        img.data()[i + 2] = 128;
        img.data()[i + 3] = 255;
    }
    return img;
}

TEST_CASE("Integration: CPUSingle and CPUMulti agree on all filters") {
    Image input = make_gradient();
    CPUSingleBackend single;
    CPUMultiBackend  multi;

    std::vector<std::unique_ptr<IFilter>> filters;
    filters.push_back(std::make_unique<Grayscale>());
    filters.push_back(std::make_unique<GaussianBlur>());
    filters.push_back(std::make_unique<SobelEdge>());
    filters.push_back(std::make_unique<BilateralFilter>());
    filters.push_back(std::make_unique<HistogramEqualization>());

    for (auto& f : filters) {
        auto r1 = single.run(*f, input);
        auto r2 = multi.run(*f, input);
        INFO("Filter: " << f->name());
        CHECK(r1.output.width() == r2.output.width());
        CHECK(r1.output.height() == r2.output.height());
        // Outputs must be pixel-identical
        CHECK(std::memcmp(r1.output.data(), r2.output.data(),
            r1.output.size()) == 0);
    }
}

TEST_CASE("Integration: Benchmark prints table without crashing") {
    Image input = make_gradient(32, 32);
    Grayscale f;
    CPUSingleBackend b;
    auto result = b.run(f, input);

    Benchmark bench;
    bench.add(result);

    std::ostringstream oss;
    bench.print(oss, "cpu_single");
    std::string table = oss.str();
    CHECK(table.find("grayscale") != std::string::npos);
    CHECK(table.find("cpu_single") != std::string::npos);
    CHECK(table.find("1.0x") != std::string::npos);
}
