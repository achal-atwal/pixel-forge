#pragma once
#include "core/IBackend.hpp"

class CPUSingleBackend : public IBackend {
public:
    std::string     name()      const override;
    bool            available() const override;
    BenchmarkResult run(const IFilter& filter, const Image& input) const override;
};
