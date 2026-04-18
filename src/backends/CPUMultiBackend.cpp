#include "backends/CPUMultiBackend.hpp"
#include "core/parallel.hpp"
#include <chrono>
#include <iostream>

std::string CPUMultiBackend::name()      const { return "cpu_multi"; }
bool        CPUMultiBackend::available() const { return true; }

BenchmarkResult CPUMultiBackend::run(const IFilter& filter, const Image& input) const {
    BenchmarkResult result;
    result.backend_name = name();
    result.filter_name = filter.name();
    result.output = Image(input.width(), input.height(), input.channels());

    std::cout << "[cpu_multi] "
#ifdef USE_OPENMP
        << "openmp"
#else
        << "std::thread"
#endif
        << "  threads=" << parallel_thread_count() << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    filter.apply_parallel(input.data(), result.output.data(),
        input.width(), input.height(), input.channels());
    auto t1 = std::chrono::high_resolution_clock::now();

    result.elapsed_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
}
