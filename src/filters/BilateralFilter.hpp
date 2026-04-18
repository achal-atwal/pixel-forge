#pragma once
#include "core/IFilter.hpp"

class BilateralFilter : public IFilter {
public:
    std::string name() const override;
    void apply(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;
    void apply_parallel(const uint8_t* in, uint8_t* out, int w, int h, int ch) const override;

private:
    static constexpr float kSigmaS = 5.0f;   // spatial sigma
    static constexpr float kSigmaR = 40.0f;  // range sigma
    static constexpr int   kRadius = 7;      // 15x15 window
};
