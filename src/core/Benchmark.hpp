#pragma once
#include "IBackend.hpp"
#include <vector>
#include <string>
#include <ostream>

class Benchmark {
public:
    void add(const BenchmarkResult& result);

    // Compute speedup relative to baseline_backend and print table to os.
    void print(std::ostream& os, const std::string& baseline_backend) const;

    // Write results as CSV (filter_name,backend_name,elapsed_ms,speedup) to os.
    void to_csv(std::ostream& os, const std::string& baseline_backend) const;

private:
    std::vector<BenchmarkResult> results_;
};
