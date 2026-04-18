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

private:
    std::vector<BenchmarkResult> results_;
};
