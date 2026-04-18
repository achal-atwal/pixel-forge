#include <catch2/catch_test_macros.hpp>
#include "backends/CPUSingleBackend.hpp"
#include "backends/CPUMultiBackend.hpp"
#include "core/IFilter.hpp"
#include "core/Image.hpp"
#include <cstring>

// Filter that sets every pixel to (10,20,30,255)
struct FlatFilter : IFilter {
    std::string name() const override { return "flat"; }
    void apply(const uint8_t*, uint8_t* out, int w, int h, int ch) const override {
        for (int i = 0; i < w * h; i++) {
            out[i * ch + 0] = 10; out[i * ch + 1] = 20; out[i * ch + 2] = 30; out[i * ch + 3] = 255;
        }
    }
};

static Image make_image(int w, int h) {
    Image img(w, h, 4);
    std::memset(img.data(), 0, img.size());
    return img;
}

TEST_CASE("CPUSingleBackend: name and available") {
    CPUSingleBackend b;
    CHECK(b.name() == "cpu_single");
    CHECK(b.available() == true);
}

TEST_CASE("CPUSingleBackend: run produces correct output") {
    CPUSingleBackend b;
    FlatFilter f;
    Image in = make_image(8, 8);
    auto result = b.run(f, in);
    CHECK(result.backend_name == "cpu_single");
    CHECK(result.filter_name == "flat");
    CHECK(result.output.width() == 8);
    CHECK(result.output.height() == 8);
    CHECK(result.output.data()[0] == 10);
    CHECK(result.output.data()[1] == 20);
    CHECK(result.output.data()[2] == 30);
    CHECK(result.output.data()[3] == 255);
    CHECK(result.elapsed_ms >= 0.0f);
}

TEST_CASE("CPUMultiBackend: output matches CPUSingle") {
    CPUSingleBackend single;
    CPUMultiBackend  multi;
    FlatFilter f;
    Image in = make_image(64, 64);
    auto r1 = single.run(f, in);
    auto r2 = multi.run(f, in);
    CHECK(r2.backend_name == "cpu_multi");
    // pixel-identical output
    CHECK(std::memcmp(r1.output.data(), r2.output.data(), r1.output.size()) == 0);
}
