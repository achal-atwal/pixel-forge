#pragma once
#include "core/IBackend.hpp"

class CPUMultiBackend : public IBackend {
public:
    std::string     name()      const override;
    bool            available() const override;
    BenchmarkResult run(const IFilter& filter, const Image& input) const override;
};
