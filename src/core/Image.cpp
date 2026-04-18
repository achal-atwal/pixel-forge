#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "core/Image.hpp"
#include <stdexcept>
#include <string>

Image::Image(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0)
        throw std::invalid_argument(
            "Image dimensions must be positive, got: " +
            std::to_string(width) + "x" + std::to_string(height) +
            "x" + std::to_string(channels));
    width_ = width;
    height_ = height;
    channels_ = channels;
    pixels_.assign(static_cast<size_t>(width) * height * channels, 0);
}

Image Image::load(const std::string& path) {
    int w, h, ch;
    uint8_t* raw = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!raw)
        throw std::runtime_error("Failed to load image: " + path +
            " — " + stbi_failure_reason());
    Image img(w, h, 4);
    std::copy(raw, raw + w * h * 4, img.pixels_.begin());
    stbi_image_free(raw);
    return img;
}

void Image::save(const std::string& path) const {
    if (empty()) throw std::runtime_error("Cannot save empty image");
    int ok = 0;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".png")
        ok = stbi_write_png(path.c_str(), width_, height_, channels_,
            pixels_.data(), width_ * channels_);
    else if ((path.size() >= 4 && path.substr(path.size() - 4) == ".jpg") ||
        (path.size() >= 5 && path.substr(path.size() - 5) == ".jpeg"))
        ok = stbi_write_jpg(path.c_str(), width_, height_, channels_,
            pixels_.data(), 90);
    else
        throw std::runtime_error("Unsupported output format (use .png or .jpg): " + path);
    if (!ok) throw std::runtime_error("Failed to write image: " + path);
}

Image Image::clone() const {
    Image copy(width_, height_, channels_);
    copy.pixels_ = pixels_;
    return copy;
}
