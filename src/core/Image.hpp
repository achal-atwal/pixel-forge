#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

class Image {
public:
    Image() = default;
    Image(int width, int height, int channels);

    static Image load(const std::string& path);   // throws std::runtime_error on failure
    void save(const std::string& path) const;      // throws std::runtime_error on failure

    Image clone() const;

    uint8_t* data() { return pixels_.data(); }
    const uint8_t* data() const { return pixels_.data(); }
    int    width()    const { return width_; }
    int    height()   const { return height_; }
    int    channels() const { return channels_; }
    size_t size()     const { return pixels_.size(); }
    bool   empty()    const { return pixels_.empty(); }

private:
    std::vector<uint8_t> pixels_;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
};
