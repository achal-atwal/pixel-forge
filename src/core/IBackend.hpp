#pragma once
#include "Image.hpp"
#include "IFilter.hpp"
#include <string>

struct BenchmarkResult {
    Image       output;
    float       elapsed_ms = 0.0f;
    std::string backend_name;
    std::string filter_name;
};

class IBackend {
public:
    virtual std::string name()      const = 0;
    virtual bool        available() const = 0;
    virtual BenchmarkResult run(const IFilter& filter, const Image& input) const = 0;
    virtual ~IBackend() = default;
};
