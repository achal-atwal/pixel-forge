#pragma once
#ifdef USE_METAL
#include "core/IBackend.hpp"
#include <memory>

struct MetalBackendImpl;  // Obj-C++ pimpl — keeps Metal headers out of .hpp

class MetalBackend : public IBackend {
public:
    MetalBackend();
    ~MetalBackend();
    std::string     name()      const override;
    bool            available() const override;
    BenchmarkResult run(const IFilter& filter, const Image& input) const override;

private:
    std::unique_ptr<MetalBackendImpl> impl_;
};
#endif // USE_METAL
