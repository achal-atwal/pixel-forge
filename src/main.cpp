#include "core/Image.hpp"
#include "core/FilterRegistry.hpp"
#include "core/BackendRegistry.hpp"
#include "core/Benchmark.hpp"
#include "filters/Grayscale.hpp"
#include "filters/GaussianBlur.hpp"
#include "filters/SobelEdge.hpp"
#include "filters/BilateralFilter.hpp"
#include "filters/HistogramEqualization.hpp"
#include "filters/KuwaharaFilter.hpp"
#include "backends/CPUSingleBackend.hpp"
#include "backends/CPUMultiBackend.hpp"
#if defined(USE_METAL)
#include "backends/MetalBackend.hpp"
#endif
#if defined(USE_CUDA)
#include "backends/CUDABackend.hpp"
#endif

#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <input.png|jpg> [options]\n"
        << "  --filter <name>       Run one filter only (default: all)\n"
        << "  --output-dir <path>   Output directory (default: ./output)\n"
        << "  --baseline <backend>  Speedup reference (default: cpu_single)\n"
        << "  --list-filters        Print filter names and exit\n"
        << "  --list-backends       Print backend names and exit\n";
}

int main(int argc, char** argv) {
    // Populate registries
    FilterRegistry filters;
    filters.add(std::make_unique<Grayscale>());
    filters.add(std::make_unique<GaussianBlur>());
    filters.add(std::make_unique<SobelEdge>());
    filters.add(std::make_unique<BilateralFilter>());
    filters.add(std::make_unique<HistogramEqualization>());
    filters.add(std::make_unique<KuwaharaFilter>());

    BackendRegistry backends;
    backends.add(std::make_unique<CPUSingleBackend>());
    backends.add(std::make_unique<CPUMultiBackend>());
#ifdef USE_METAL
    backends.add(std::make_unique<MetalBackend>());
#endif
#ifdef USE_CUDA
    backends.add(std::make_unique<CUDABackend>());
#endif

    // Parse arguments
    std::string input_path, filter_name, output_dir = "output", baseline = "cpu_single";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--filter" && i + 1 < argc) { filter_name = argv[++i]; }
        else if (arg == "--output-dir" && i + 1 < argc) { output_dir = argv[++i]; }
        else if (arg == "--baseline" && i + 1 < argc) { baseline = argv[++i]; }
        else if (arg == "--list-filters") {
            for (auto& n : filters.names()) std::cout << n << "\n";
            return 0;
        }
        else if (arg == "--list-backends") {
            for (auto* b : backends.available())
                std::cout << b->name() << "\n";
            return 0;
        }
        else if (arg[0] != '-') { input_path = arg; }
        else { std::cerr << "Unknown option: " << arg << "\n"; usage(argv[0]); return 1; }
    }

    if (input_path.empty()) { usage(argv[0]); return 1; }

    // Validate inputs
    auto ext = fs::path(input_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") {
        std::cerr << "Error: only .png and .jpg inputs supported\n";
        return 1;
    }

    if (!filter_name.empty() && !filters.get(filter_name)) {
        std::cerr << "Unknown filter: " << filter_name
            << ". Use --list-filters.\n"; return 1;
    }

    if (!backends.get(baseline)) {
        std::cerr << "Unknown baseline backend: " << baseline
            << ". Use --list-backends.\n"; return 1;
    }

    // Load image
    Image input;
    try { input = Image::load(input_path); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    std::cout << "Loaded: " << input_path
        << " (" << input.width() << "x" << input.height() << ")\n";

    fs::create_directories(output_dir);
    std::string stem = fs::path(input_path).stem().string();

    // Run benchmarks
    Benchmark bench;
    auto active_filters = filter_name.empty() ? filters.names()
        : std::vector<std::string>{ filter_name };
    auto active_backends = backends.available();

    for (auto& fn : active_filters) {
        IFilter* f = filters.get(fn);
        if (!f) continue;  // defensive: should never happen
        for (auto* b : active_backends) {
            std::cout << "  " << fn << " / " << b->name() << " ... " << std::flush;
            try {
                auto result = b->run(*f, input);
                bench.add(result);

                std::string out_path = output_dir + "/" + stem + "_" +
                    fn + "_" + b->name() + ".png";
                result.output.save(out_path);
                std::cout << result.elapsed_ms << " ms  →  " << out_path << "\n";
            }
            catch (const std::exception& e) {
                std::cerr << "FAILED: " << e.what() << "\n";
            }
        }
    }

    bench.print(std::cout, baseline);
    return 0;
}
