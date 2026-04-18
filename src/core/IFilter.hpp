#pragma once
#include <cstdint>
#include <string>

class IFilter {
public:
    virtual std::string name() const = 0;

    // Single-threaded — called by CPUSingleBackend
    virtual void apply(const uint8_t* in, uint8_t* out,
        int width, int height, int channels) const = 0;

    // Multi-threaded — called by CPUMultiBackend.
    // Default delegates to apply(); filters override with OpenMP loops.
    virtual void apply_parallel(const uint8_t* in, uint8_t* out,
        int width, int height, int channels) const {
        apply(in, out, width, height, channels);
    }

    virtual ~IFilter() = default;
};
