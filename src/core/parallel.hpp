#pragma once
#include <thread>
#include <vector>
#include <algorithm>
#ifdef USE_OPENMP
#include <omp.h>
#endif

// Returns the thread count that parallel_for will use.
inline int parallel_thread_count() {
#ifdef USE_OPENMP
    return omp_get_max_threads();
#else
    int n = (int)std::thread::hardware_concurrency();
    return n > 0 ? n : 1;
#endif
}

// Calls f(i) for i in [0, n) in parallel.
// OpenMP path: schedule(static), one iteration per index.
// std::thread path: N threads each own a contiguous chunk.
template<typename Func>
inline void parallel_for(int n, Func&& f) {
    if (n <= 0) return;
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) f(i);
#else
    int nthreads = std::min(parallel_thread_count(), n);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; t++) {
        int lo = t * n / nthreads;
        int hi = (t + 1) * n / nthreads;
        threads.emplace_back([&f, lo, hi]() {
            for (int i = lo; i < hi; i++) f(i);
            });
    }
    for (auto& th : threads) th.join();
#endif
}
