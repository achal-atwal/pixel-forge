#include "core/Benchmark.hpp"
#include <iomanip>
#include <map>
#include <iostream>

void Benchmark::add(const BenchmarkResult& result) {
    results_.push_back(result);
}

void Benchmark::print(std::ostream& os, const std::string& baseline) const {
    // Build baseline map: filter_name -> elapsed_ms for baseline backend
    std::map<std::string, float> base_ms;
    for (auto& r : results_)
        if (r.backend_name == baseline)
            base_ms[r.filter_name] = r.elapsed_ms;

    const int w0 = 24, w1 = 16, w2 = 14, w3 = 10;
    os << "\n"
        << std::left << std::setw(w0) << "Filter"
        << std::setw(w1) << "Backend"
        << std::right << std::setw(w2) << "Time (ms)"
        << std::setw(w3) << "Speedup" << "\n";
    os << std::string(w0 + w1 + w2 + w3, '-') << "\n";

    std::string last_filter;
    for (auto& r : results_) {
        if (r.filter_name != last_filter && !last_filter.empty())
            os << "\n";
        last_filter = r.filter_name;

        float base = base_ms.count(r.filter_name) ? base_ms.at(r.filter_name) : r.elapsed_ms;
        float speedup = (r.elapsed_ms > 0) ? (base / r.elapsed_ms) : 0.f;

        os << std::left << std::setw(w0) << r.filter_name
            << std::setw(w1) << r.backend_name
            << std::right << std::setw(w2 - 3) << std::fixed << std::setprecision(2)
            << r.elapsed_ms << " ms"
            << std::setw(w3 - 1) << std::setprecision(1) << speedup << "x\n";
    }
    os << "\n";
}

void Benchmark::to_csv(std::ostream& os, const std::string& baseline) const {
    std::map<std::string, float> base_ms;
    for (auto& r : results_)
        if (r.backend_name == baseline)
            base_ms[r.filter_name] = r.elapsed_ms;

    os << "filter_name,backend_name,elapsed_ms,speedup\n";
    for (auto& r : results_) {
        float base = base_ms.count(r.filter_name) ? base_ms.at(r.filter_name) : r.elapsed_ms;
        float speedup = (r.elapsed_ms > 0) ? (base / r.elapsed_ms) : 0.f;
        os << r.filter_name << ","
            << r.backend_name << ","
            << std::fixed << std::setprecision(2) << r.elapsed_ms << ","
            << std::setprecision(4) << speedup << "\n";
    }
}
