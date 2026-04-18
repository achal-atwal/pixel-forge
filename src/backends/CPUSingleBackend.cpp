#include "backends/CPUSingleBackend.hpp"
#include <chrono>

std::string CPUSingleBackend::name()      const { return "cpu_single"; }
bool        CPUSingleBackend::available() const { return true; }

BenchmarkResult CPUSingleBackend::run(const IFilter& filter, const Image& input) const {
    BenchmarkResult result;
    result.backend_name = name();
    result.filter_name = filter.name();
    result.output = Image(input.width(), input.height(), input.channels());

    auto t0 = std::chrono::high_resolution_clock::now();
    filter.apply(input.data(), result.output.data(),
        input.width(), input.height(), input.channels());
    auto t1 = std::chrono::high_resolution_clock::now();

    result.elapsed_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}
