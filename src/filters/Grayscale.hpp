#pragma once
#include "core/IFilter.hpp"

class Grayscale : public IFilter {
public:
    std::string name() const override;
    void apply(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;
    void apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;
};
