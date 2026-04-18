#pragma once
#include "core/IFilter.hpp"

class KuwaharaFilter : public IFilter {
public:
    std::string name() const override;
    void apply(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;
    void apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;

private:
    static constexpr int kRadius = 5;  // 6x6 quadrants = 36 pixels each
};
