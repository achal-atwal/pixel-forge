#pragma once
#ifdef USE_CUDA
#include "core/IBackend.hpp"

class CUDABackend : public IBackend {
public:
    CUDABackend();
    std::string     name()      const override;
    bool            available() const override;
    BenchmarkResult run(const IFilter& filter, const Image& input) const override;

private:
    bool available_;
};
#endif
