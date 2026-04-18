#pragma once
#include "core/IFilter.hpp"

class GaussianBlur : public IFilter {
public:
    std::string name() const override;
    void apply(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;
    void apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;

private:
    // 1-D Gaussian kernel, σ≈2.0, radius 3
    static constexpr float kKernel[7] = { 0.0702f, 0.1311f, 0.1907f, 0.2161f, 0.1907f, 0.1311f, 0.0702f };
};
