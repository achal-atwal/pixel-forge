#include <catch2/catch_test_macros.hpp>
#include "core/Image.hpp"
#include <cstring>

TEST_CASE("Image: construct with dimensions") {
    Image img(4, 3, 4);
    CHECK(img.width() == 4);
    CHECK(img.height() == 3);
    CHECK(img.channels() == 4);
    CHECK(img.size() == 4u * 3u * 4u);
    CHECK_FALSE(img.empty());
}

TEST_CASE("Image: default-constructed is empty") {
    Image img;
    CHECK(img.empty());
    CHECK(img.width() == 0);
}

TEST_CASE("Image: clone produces independent copy") {
    Image img(2, 2, 4);
    std::memset(img.data(), 128, img.size());
    Image copy = img.clone();
    CHECK(copy.width() == img.width());
    CHECK(copy.size() == img.size());
    // Mutate original — copy must not change
    img.data()[0] = 255;
    CHECK(copy.data()[0] == 128);
}

TEST_CASE("Image: load rejects non-existent file") {
    CHECK_THROWS(Image::load("nonexistent_file_xyz.png"));
}
