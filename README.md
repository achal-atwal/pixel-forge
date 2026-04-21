# pixel-forge

A cross-platform C++ tool that applies image filters using CPU (single-threaded and multi-threaded via OpenMP) and GPU (Metal on macOS ARM, CUDA on NVIDIA Windows). After processing, it prints a benchmark table comparing every backend side by side.

The project is designed to demonstrate the real-world performance difference between CPU and GPU image processing, with a clean abstraction layer that makes it easy to add new filters or target new hardware.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Image Formats and Color Model](#image-formats-and-color-model)
- [Filters](#filters)
- [Backends](#backends)
- [Build](#build)
  - [Xcode](#building-with-xcode)
  - [Visual Studio](#building-with-visual-studio)
- [Usage](#usage)
- [Example Output](#example-output)
- [Adding a New Filter](#adding-a-new-filter)
- [Adding a New Backend](#adding-a-new-backend)
- [Dependencies](#dependencies)
- [Future Roadmap](#future-roadmap)

---

## How It Works

1. You provide an input image (PNG or JPEG).
2. pixel-forge loads it into an RGBA buffer in memory.
3. For each filter × backend combination, it runs the filter through that backend, times it, and saves the output image.
4. After all combinations complete, it prints a benchmark table showing elapsed time and speedup relative to a configurable baseline backend.

```
input.jpg  ──►  [Filter: grayscale]  ──►  cpu_single  ──►  out/input_grayscale_cpu_single.png  (12.3 ms)
                                     ──►  cpu_multi   ──►  out/input_grayscale_cpu_multi.png   (3.1 ms)
                                     ──►  metal       ──►  out/input_grayscale_metal.png        (0.8 ms)
               [Filter: gaussian_blur] ──► ...
```

---

## Architecture

pixel-forge uses a **backend-centric** design. Filters define the pixel math; backends own the execution strategy. The two sides are completely independent.

```
┌─────────────────────────────────────────────────────────────────────┐
│  FilterRegistry               BackendRegistry                       │
│  ─────────────                ──────────────                        │
│  grayscale        ◄──────►   cpu_single                            │
│  gaussian_blur               cpu_multi                              │
│  sobel_edge                  metal  (macOS ARM only)                │
│  bilateral_filter            cuda   (NVIDIA only)                   │
│  histogram_eq                                                       │
│  kuwahara                                                           │
└─────────────────────────────────────────────────────────────────────┘
         │                             │
         ▼                             ▼
   IFilter::apply()            IBackend::run(filter, image)
   IFilter::apply_parallel()   └─► BenchmarkResult { output, elapsed_ms }
```

### Key interfaces

**`IFilter`** (`src/core/IFilter.hpp`)

```cpp
class IFilter {
public:
    virtual std::string name() const = 0;

    // Called by CPUSingleBackend — sequential loop
    virtual void apply(const uint8_t* in, uint8_t* out,
                       int w, int h, int channels) const = 0;

    // Called by CPUMultiBackend — override with OpenMP loop.
    // Default falls back to apply().
    virtual void apply_parallel(const uint8_t* in, uint8_t* out,
                                int w, int h, int channels) const;
};
```

**`IBackend`** (`src/core/IBackend.hpp`)

```cpp
struct BenchmarkResult {
    Image       output;
    float       elapsed_ms;
    std::string backend_name;
    std::string filter_name;
};

class IBackend {
public:
    virtual std::string     name()      const = 0;
    virtual bool            available() const = 0;
    virtual BenchmarkResult run(const IFilter&, const Image&) const = 0;
};
```

### Execution paths by backend

| Backend | How it runs a filter |
|---|---|
| `CPUSingleBackend` | Calls `filter.apply()` — plain sequential C++ |
| `CPUMultiBackend` | Calls `filter.apply_parallel()` — OpenMP parallel-for loop |
| `MetalBackend` | Dispatches a Metal compute kernel keyed by `filter.name()`, bypasses `IFilter::apply()` entirely |
| `CUDABackend` | Calls a CUDA launcher keyed by `filter.name()`, bypasses `IFilter::apply()` entirely |

GPU backends **do not call** `IFilter::apply()`. They dispatch their own shaders and only use `filter.name()` to look up the right kernel.

### Registries

Both `FilterRegistry` and `BackendRegistry` preserve insertion order (important for consistent benchmark output) and throw `std::invalid_argument` on duplicate names.

### Benchmark

`Benchmark` collects `BenchmarkResult` values and provides two output methods:

- `print(os, baseline)` — formatted table with speedup columns, written to any `std::ostream`
- `to_csv(os, baseline)` — CSV with columns `filter_name,backend_name,elapsed_ms,speedup`, written to any `std::ostream`

Speedup is computed as `baseline_elapsed / this_elapsed`. The baseline defaults to `cpu_single` and is configurable with `--baseline`.

---

## Image Formats and Color Model

### Supported input formats

| Format | Extensions | Notes |
|---|---|---|
| PNG | `.png` | Lossless, supports transparency (RGBA) |
| JPEG | `.jpg`, `.jpeg` | Lossy compression, no transparency |

Any other extension is rejected with an error at startup.

### Internal color model

All images are loaded and processed as **RGBA, 8 bits per channel** (4 bytes per pixel), regardless of the source format:

- Channel 0 — **R** (red), 0–255
- Channel 1 — **G** (green), 0–255
- Channel 2 — **B** (blue), 0–255
- Channel 3 — **A** (alpha), 0–255 — preserved by all filters, never modified

This normalization is done at load time by `stb_image` (`stbi_load(..., 4)`). A grayscale source becomes `R=G=B=value, A=255`. A JPEG (no native alpha) becomes `A=255`.

### Output

Output images are always saved as **PNG** regardless of the input format, using `stb_image_write`. JPEG quality is 90 when saving `.jpg`/`.jpeg` (used when saving via `Image::save()` with those extensions).

---

## Filters

All filters receive and produce RGBA images. RGB channels are modified; **alpha is always preserved unchanged**.

### `grayscale`

Converts each pixel to luma using the ITU-R BT.601 formula and writes it to all three RGB channels.

```
luma = 0.299·R + 0.587·G + 0.114·B
output pixel = (luma, luma, luma, A)
```

- **GPU characteristic:** Memory-bound — one read, one write per pixel, no neighborhood access.
- **Parallel strategy:** Embarrassingly parallel; each pixel is independent. OpenMP `schedule(static)`.

---

### `gaussian_blur`

Separable 5×5 Gaussian blur. Runs two passes: horizontal then vertical. This is mathematically equivalent to a 2D convolution with a 5×5 Gaussian kernel but requires only O(5·w·h) operations instead of O(25·w·h).

**1-D kernel** (σ ≈ 1.0, radius 2):

```
k = [ 0.0545,  0.2442,  0.4026,  0.2442,  0.0545 ]
```

Border pixels use **clamp-to-edge** padding — the nearest valid pixel is repeated.

- **GPU characteristic:** Local convolution — each output pixel requires 5 reads (per pass).
- **Parallel strategy:** Each row is independent. OpenMP `schedule(static)`. Two separate parallel-for blocks (H pass then V pass) with an implicit barrier between them.

---

### `sobel_edge`

Edge detection using the Sobel operator. Computes a horizontal gradient (Gx) and vertical gradient (Gy) from the luma of the input, then outputs the gradient magnitude in all three RGB channels.

```
luma(x,y) = 0.299·R + 0.587·G + 0.114·B

Gx = [-1  0 +1]     Gy = [-1 -2 -1]
     [-2  0 +2]          [ 0  0  0]
     [-1  0 +1]          [+1 +2 +1]

magnitude = clamp(sqrt(Gx² + Gy²), 0, 255)
output pixel = (magnitude, magnitude, magnitude, A)
```

Border pixels use **clamp-to-edge** padding. Interior of a uniform image produces magnitude 0 (no edges).

- **GPU characteristic:** 3×3 neighborhood read per pixel.
- **Parallel strategy:** Each row is independent. OpenMP `schedule(static)`.

---

### `bilateral_filter`

Edge-preserving smoothing filter. For each pixel, computes a weighted average of its 9×9 neighborhood. The weight of each neighbor combines a **spatial Gaussian** (distance from center) and a **range Gaussian** (color difference from center). Pixels with similar color but far away are downweighted spatially; pixels with very different color but close are downweighted by range. This preserves edges while smoothing uniform regions.

**Parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `kSigmaS` | 3.0 | Spatial Gaussian σ — how fast spatial weight falls off |
| `kSigmaR` | 30.0 | Range Gaussian σ (in RGB space, 0–255) — color similarity threshold |
| `kRadius` | 4 | Neighborhood radius → 9×9 window |

**Weight formula for neighbor at offset (dx, dy):**

```
ds = dx² + dy²                         (squared spatial distance)
dr = ΔR² + ΔG² + ΔB²                  (squared color difference)
w  = exp(-ds / (2·σs²)  -  dr / (2·σr²))
```

Output = weighted average of neighbors / sum of weights.

- **GPU characteristic:** Compute-heavy, data-dependent weights — shows high GPU speedup, second only to `kuwahara`.
- **Parallel strategy:** Each row is independent but compute-uneven (edge pixels have different weight distributions). OpenMP `schedule(dynamic, 4)`.

---

### `histogram_eq`

Per-channel histogram equalization. Stretches the tonal range of each color channel independently using the Cumulative Distribution Function (CDF). Dark images become brighter; washed-out images gain contrast.

**Algorithm per channel:**

1. Build a 256-bin histogram from all pixel values in that channel.
2. Compute the CDF (running total of the histogram).
3. Find `cdf_min` — the CDF value at the first non-zero bin.
4. Build a lookup table: `lut[v] = round((cdf[v] - cdf_min) / (total - cdf_min) * 255)`.
5. Remap every pixel through the LUT.

A uniform image (all pixels same color) produces `denom = 0`, which is handled by passthrough (`lut[v] = v`).

The GPU implementation (Metal and CUDA) precomputes the LUTs on the CPU, uploads them as small 256-byte buffers, then parallelizes only the remap pass on the GPU — since histogram reduction requires global reduction which is not trivially parallelizable in a simple kernel.

- **GPU characteristic:** Global reduction (histogram) + per-pixel remap. Remap phase is memory-bound.
- **Parallel strategy:** Histogram build is sequential (serial reduction). Remap loop is parallelized. OpenMP `schedule(static)`.

---

### `kuwahara`

Edge-preserving artistic filter that produces an oil-painting effect. For each output pixel, the surrounding area is divided into four overlapping quadrants. The output is the **mean color of the quadrant with the lowest luminance variance** — smooth regions are averaged, but the quadrant that straddles an edge is always discarded in favor of the quieter side, so edges remain sharp.

**Parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `kRadius` | 5 | Quadrant radius → four overlapping 6×6 regions (36 pixels each) |

**Algorithm per pixel:**

```
For each of 4 quadrants Q ∈ {top-left, top-right, bottom-left, bottom-right}:
    dx ∈ [-r, 0] or [0, r]    dy ∈ [-r, 0] or [0, r]
    For each neighbor (nx, ny) in Q (clamped to image bounds):
        accumulate: sum_R, sum_G, sum_B, sum_luma, sum_luma²
    mean_luma = sum_luma / count
    variance  = sum_luma² / count − mean_luma²
    mean_RGB  = (sum_R, sum_G, sum_B) / count

output pixel = mean_RGB of the quadrant with minimum variance
```

Quadrants overlap along their shared edges (the center pixel belongs to all four). Each quadrant covers `(r+1)² = 36` pixels at r=5, for a total neighborhood of ~11×11 pixels.

- **GPU characteristic:** High arithmetic intensity — ~144 multiply-adds per output pixel (4 quadrants × 36 neighbors). Among the most compute-bound filters in this project and the best showcase of GPU throughput over CPU.
- **Parallel strategy:** Each pixel is fully independent. OpenMP `schedule(static)` per row.

---

## Backends

### `cpu_single`

**File:** `src/backends/CPUSingleBackend.cpp`

The simplest backend — a direct, sequential call to `filter.apply()` with wall-clock timing around it.

**Execution flow:**

```
run(filter, input)
  │
  ├─ allocate output Image (same dimensions as input)
  ├─ t0 = high_resolution_clock::now()
  ├─ filter.apply(input.data(), output.data(), w, h, channels)   ← single thread
  ├─ t1 = high_resolution_clock::now()
  └─ return BenchmarkResult { output, elapsed_ms = (t1-t0) }
```

- **Timing:** `std::chrono::high_resolution_clock` wraps the `apply()` call directly. This measures pure filter computation time on the CPU.
- **Always available** — no dependencies, no hardware requirement.
- Serves as the default **speedup baseline** (`--baseline cpu_single`).

---

### `cpu_multi`

**File:** `src/backends/CPUMultiBackend.cpp`

Identical structure to `cpu_single`, but calls `filter.apply_parallel()` instead of `filter.apply()`. Each filter's `apply_parallel()` adds `#pragma omp parallel for` to its inner loop, distributing rows across CPU threads managed by OpenMP.

**Execution flow:**

```
run(filter, input)
  │
  ├─ allocate output Image
  ├─ t0 = high_resolution_clock::now()
  ├─ filter.apply_parallel(input.data(), output.data(), w, h, channels)
  │    └─ OpenMP splits the pixel loop across N threads (N = CPU core count)
  ├─ t1 = high_resolution_clock::now()
  └─ return BenchmarkResult { output, elapsed_ms = (t1-t0) }
```

- **Thread count:** Controlled by the `OMP_NUM_THREADS` environment variable. Defaults to the number of logical CPU cores reported by the OS.

  ```bash
  OMP_NUM_THREADS=4 ./build/pixel_forge photo.jpg
  ```
- **Compiled in** when CMake finds OpenMP (`find_package(OpenMP)`). If OpenMP is absent, `apply_parallel()` falls back to calling `apply()` — still correct, just not parallel.
- Each filter chooses its OpenMP schedule independently:
  - `grayscale`, `gaussian_blur`, `sobel_edge`, `histogram_eq` (remap pass): `schedule(static)` — uniform work per row.
  - `bilateral_filter`: `schedule(dynamic, 4)` — work per row varies by image content.

---

### `metal` (macOS ARM only)

**Files:** `src/backends/MetalBackend.hpp`, `src/backends/MetalBackend.mm`

Uses Apple's **Metal compute API** to run filter kernels on the Apple Silicon GPU. The backend uses the **pimpl pattern** — all Objective-C++ and Metal types are hidden inside `MetalBackendImpl`, keeping Metal headers out of the `.hpp` and away from the rest of the C++ codebase.

#### Initialization (constructor, runs once)

```
MetalBackend()
  │
  ├─ MTLCreateSystemDefaultDevice()       — get the default GPU
  ├─ [device newCommandQueue]             — create a command queue
  ├─ [device newLibraryWithSource: kMetalSrc]
  │    └─ kMetalSrc is a string literal in MetalBackend.mm containing all
  │       five Metal kernel functions; Metal compiles it at runtime via the
  │       Metal compiler embedded in macOS
  ├─ for each kernel name in [grayscale, gaussian_blur, sobel_edge,
  │                            bilateral_filter, histogram_eq, kuwahara]:
  │    ├─ [lib newFunctionWithName: name]
  │    └─ [device newComputePipelineStateWithFunction: func]
  │         └─ stores compiled MTLComputePipelineState in pipelines map
  └─ ready = (pipelines.size() == 6)     — true only if all 6 compiled OK
```

If any kernel fails to compile, a warning is logged and `available()` returns `false` for the entire backend.

#### Per-filter run

```
run(filter, input)
  │
  ├─ validate: ready=true, filter.name() exists in pipelines map
  │
  ├─ [device newBufferWithBytes: input.data()]   ← buf_in  (copies pixel data)
  ├─ [device newBufferWithLength: bytes]          ← buf_out (empty, GPU writes here)
  ├─ [device newBufferWithBytes: &{w,h}]          ← buf_dims (image dimensions)
  │
  ├─ [histogram_eq only]
  │    ├─ build R/G/B LUTs on CPU (256 bytes each)
  │    └─ upload as buf_lut_r/g/b (buffers 3/4/5)
  │
  ├─ [cmd computeCommandEncoder]
  │    ├─ setComputePipelineState: pipelines[filter.name()]
  │    ├─ setBuffer: buf_in  at index 0
  │    ├─ setBuffer: buf_out at index 1
  │    ├─ setBuffer: buf_dims at index 2
  │    ├─ [histogram_eq] setBuffer: lut_r/g/b at index 3/4/5
  │    └─ dispatchThreads: grid={(w+15)/16·16, (h+15)/16·16, 1}
  │                         threadsPerThreadgroup:{16, 16, 1}
  │
  ├─ addScheduledHandler → t0 = CFAbsoluteTimeGetCurrent()
  ├─ addCompletedHandler → t1 = CFAbsoluteTimeGetCurrent()
  ├─ [cmd commit]
  ├─ [cmd waitUntilCompleted]              ← blocks until GPU finishes
  │
  ├─ memcpy [buf_out contents] → result.output   ← read back GPU output
  └─ return BenchmarkResult { output, elapsed_ms = (t1-t0)*1000 }
```

**Threadgroup layout:** Each Metal thread processes exactly one pixel. The GPU grid is rounded up to the nearest 16×16 multiple; out-of-bounds threads exit immediately via a bounds check at the top of every kernel. For a 1920×1080 image, the grid is 1920×1088 (1088 = ⌈1080/16⌉×16), dispatching 2,088,960 threads simultaneously.

**Memory model:** On Apple Silicon, CPU and GPU share the same physical RAM (unified memory). `MTLResourceStorageModeShared` buffers are directly accessible by both sides with no explicit DMA transfer — the `newBufferWithBytes:` call copies the pixel data into a Metal-managed allocation, but there is no PCIe bus transfer as on discrete GPUs.

**Gaussian blur note:** The Metal kernel applies the Gaussian as a **2D convolution** (not separable two-pass) — it multiplies `k[dx+2] * k[dy+2]` directly. This produces identical output to the CPU separable approach but does 25 multiply-adds per pixel rather than 10.

---

### `cuda` (NVIDIA Windows only)

**Files:** `src/backends/CUDABackend.hpp`, `src/backends/CUDABackend.cpp`, `src/shaders/filters.cu`

Uses the **CUDA runtime API** to run filter kernels on an NVIDIA GPU. Kernels are compiled ahead-of-time by `nvcc` into the binary (unlike Metal which compiles shaders at runtime). The `.cu` file contains the kernels and `extern "C"` launcher functions; `CUDABackend.cpp` calls those launchers.

#### Initialization (constructor, runs once)

```
CUDABackend()
  └─ cudaGetDeviceCount(&count)
       available_ = (result == cudaSuccess && count > 0)
```

No kernel compilation happens at init — kernels are baked into the binary at build time by `nvcc`.

#### Per-filter run

```
run(filter, input)
  │
  ├─ validate: available_=true, filter name is one of the 6 known filters
  │
  ├─ cudaMalloc(&d_in,  bytes)       ← allocate device input buffer
  ├─ cudaMalloc(&d_out, bytes)       ← allocate device output buffer
  ├─ cudaMemcpy(d_in, input.data(), bytes, HostToDevice)   ← upload image
  │
  ├─ cudaStreamCreate(&stream)       ← create async execution stream
  ├─ cudaEventCreate(&ev0/ev1)       ← GPU-side timer events
  │
  ├─ [histogram_eq only]
  │    ├─ build R/G/B LUTs on CPU (256 bytes each)
  │    ├─ cudaMalloc + cudaMemcpy for d_lut_r/g/b   ← upload LUTs
  │
  ├─ cudaEventRecord(ev0, stream)    ← start GPU timer
  ├─ launch_<filtername>(d_in, d_out, w, h, stream)   ← enqueue kernel
  │    └─ each launcher: dim3 block(16,16), grid((w+15)/16, (h+15)/16)
  │       k_<filtername><<<grid, block, 0, stream>>>(...)
  ├─ cudaEventRecord(ev1, stream)    ← stop GPU timer
  │
  ├─ cudaStreamSynchronize(stream)   ← wait for GPU to finish
  ├─ cudaEventElapsedTime(&elapsed_ms, ev0, ev1)   ← read GPU timer
  │
  ├─ cudaMemcpy(output.data(), d_out, bytes, DeviceToHost)   ← download result
  │
  ├─ cudaFree(d_in), cudaFree(d_out)
  ├─ [histogram_eq] cudaFree(d_lut_r/g/b)
  ├─ cudaEventDestroy(ev0/ev1)
  ├─ cudaStreamDestroy(stream)
  └─ return BenchmarkResult { output, elapsed_ms }
```

**Timing accuracy:** `cudaEventRecord` inserts timestamp markers directly into the GPU command stream. `cudaEventElapsedTime` reads the delta between them after synchronization. This measures **only GPU execution time** — host↔device memory transfers are excluded from the benchmark. This is different from the Metal backend where timing includes all GPU-side work after the command buffer is scheduled.

**Memory model:** NVIDIA discrete GPUs have their own VRAM, separate from CPU RAM. `cudaMemcpy(..., HostToDevice)` performs a DMA transfer over the PCIe bus before the kernel runs, and `DeviceToHost` transfers the result back after. This transfer cost is **not included** in `elapsed_ms` — it is intentionally excluded to isolate kernel performance.

**Threadblock layout:** Same 16×16 block as Metal. The grid is `ceil(w/16) × ceil(h/16)` blocks. Out-of-bounds threads exit immediately at the top of each kernel.

---

## Build

Requirements: CMake 3.20+, a C++17 compiler.

**macOS ARM (Metal — automatic):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**Windows (NVIDIA — automatic):**

```bash
cmake -B build -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE="$PWD/build"
cmake --build build --config Release -j --parallel
```

CMake auto-detects the platform:
- Apple ARM → Metal enabled, CUDA skipped
- CUDA compiler found → CUDA enabled
- Neither → CPU-only build (cpu_single + cpu_multi)

OpenMP is detected automatically by `find_package(OpenMP)`. If not found, `cpu_multi` falls back to calling `filter.apply()` sequentially.

**Run tests:**

```bash
./build/tests
```

34 tests, ~10700 assertions covering Image, all 6 filters, both CPU backends, registries, and integration.

---

### Building with Xcode

CMake can generate a native `.xcodeproj`:

```bash
cmake -B build-xcode -G Xcode
open build-xcode/pixel_forge.xcodeproj
```

Select the `pixel_forge` or `tests` scheme in the toolbar, then **Product → Build** (⌘B) or **Product → Run** (⌘R). Set breakpoints in any `.cpp` or `.mm` file normally.

Two things to note:
- `CMAKE_BUILD_TYPE` is ignored with the Xcode generator — Xcode manages Debug/Release through scheme settings (**Product → Scheme → Edit Scheme → Run → Build Configuration**).
- The Metal shader source is embedded as a string literal in `MetalBackend.mm`. For GPU-side debugging, use Xcode's Metal Debugger: run the app, then **Debug → Capture GPU Frame** to inspect kernel execution and buffer contents.

---

### Building with Visual Studio

**Option A: Open folder (no extra step needed)**

Visual Studio 2019+ detects `CMakeLists.txt` automatically:

1. **File → Open → Folder** → select the `pixel-forge` directory
2. VS configures the project automatically
3. Select a startup item from the toolbar dropdown (`pixel_forge` or `tests`)
4. **Build → Build All** (Ctrl+Shift+B), then **Debug → Start Debugging** (F5)

Build configuration (Debug/Release) is controlled via **Project → CMake Settings**.

**Option B: Generate a `.sln` manually**

```cmd
cmake -B build-vs -G "Visual Studio 17 2022" -A x64
start build-vs\pixel_forge.sln
```

Replace `17 2022` with your installed version (`16 2019`, `15 2017`, etc.).

**Windows-specific notes:**
- Metal backend is excluded automatically — CMake detects non-Apple platform
- CUDA is picked up if `nvcc` is on `PATH`; set `CMAKE_CUDA_COMPILER` explicitly if needed
- OpenMP is bundled with MSVC — `find_package(OpenMP)` finds it without any extra install

---

## Usage

```bash
./build/pixel_forge <input.png|jpg> [options]

Options:
  --filter <name>       Run one filter only (default: all)
  --output-dir <path>   Save output images here (default: ./output)
  --baseline <backend>  Speedup reference (default: cpu_single)
  --csv <file>          Export benchmark results to a CSV file
  --list-filters        Print available filter names and exit
  --list-backends       Print available backend names and exit
```

**List filters:**

```bash
./build/pixel_forge --list-filters
# grayscale
# gaussian_blur
# sobel_edge
# bilateral_filter
# histogram_eq
# kuwahara
```

**List backends:**

```bash
./build/pixel_forge --list-backends
# cpu_single
# cpu_multi
# metal        (macOS ARM only)
# cuda         (NVIDIA only)
```

**Run all filters on an image:**

```bash
./build/pixel_forge photo.jpg --output-dir ./out
```

**Run one filter, compare speedup against cpu_multi:**

```bash
./build/pixel_forge photo.jpg --filter bilateral_filter --baseline cpu_multi
```

**Export benchmark results to CSV:**

```bash
./build/pixel_forge photo.jpg --csv results.csv
```

The CSV contains one row per filter/backend combination with columns `filter_name,backend_name,elapsed_ms,speedup`.

Output images are named `<stem>_<filter>_<backend>.png`, e.g. `photo_grayscale_metal.png`.

---

## Project Structure

```
pixel-forge/
├── CMakeLists.txt
├── README.md
├── third_party/
│   └── stb/
│       ├── stb_image.h
│       └── stb_image_write.h
├── src/
│   ├── main.cpp                         CLI entry point
│   ├── core/
│   │   ├── Image.hpp / Image.cpp        RGBA image container (load/save/clone)
│   │   ├── stb_impl.cpp                 stb_image implementation (ODR isolation)
│   │   ├── IFilter.hpp                  Abstract filter interface
│   │   ├── IBackend.hpp                 Abstract backend interface + BenchmarkResult
│   │   ├── FilterRegistry.hpp/.cpp      Insertion-ordered registry of filters
│   │   ├── BackendRegistry.hpp/.cpp     Insertion-ordered registry of backends
│   │   └── Benchmark.hpp/.cpp           Result collector + table printer
│   ├── filters/
│   │   ├── Grayscale.hpp/.cpp
│   │   ├── GaussianBlur.hpp/.cpp
│   │   ├── SobelEdge.hpp/.cpp
│   │   ├── BilateralFilter.hpp/.cpp
│   │   ├── HistogramEqualization.hpp/.cpp
│   │   └── KuwaharaFilter.hpp/.cpp
│   ├── backends/
│   │   ├── CPUSingleBackend.hpp/.cpp
│   │   ├── CPUMultiBackend.hpp/.cpp
│   │   ├── MetalBackend.hpp/.mm         (compiled only on Apple ARM)
│   │   └── CUDABackend.hpp/.cpp         (compiled only with nvcc)
│   └── shaders/
│       └── filters.cu                   CUDA kernels for all 6 filters
└── tests/
    ├── test_image.cpp                   Image class unit tests
    ├── test_filters.cpp                 Filter unit tests (all 6 filters)
    ├── test_backends.cpp                Backend unit tests (CPU backends)
    ├── test_registries.cpp              Registry unit tests
    └── test_integration.cpp             End-to-end pipeline tests
```

---

## Example Output

```
Loaded: photo.jpg (1920x1080)
  grayscale / cpu_single ... 12.3 ms  →  output/photo_grayscale_cpu_single.png
  grayscale / cpu_multi  ...  3.1 ms  →  output/photo_grayscale_cpu_multi.png
  grayscale / metal      ...  0.8 ms  →  output/photo_grayscale_metal.png
  bilateral_filter / cpu_single ... 312.4 ms  →  ...
  bilateral_filter / cpu_multi  ...  84.2 ms  →  ...
  bilateral_filter / metal      ...   6.1 ms  →  ...

Filter                  Backend              Time (ms)   Speedup
----------------------------------------------------------------
grayscale               cpu_single             12.3 ms      1.0x
grayscale               cpu_multi               3.1 ms      3.9x
grayscale               metal                   0.8 ms     15.4x

gaussian_blur           cpu_single             18.7 ms      1.0x
gaussian_blur           cpu_multi               5.2 ms      3.6x
gaussian_blur           metal                   1.1 ms     17.0x

sobel_edge              cpu_single             24.5 ms      1.0x
sobel_edge              cpu_multi               6.3 ms      3.9x
sobel_edge              metal                   1.4 ms     17.5x

bilateral_filter        cpu_single            312.4 ms      1.0x
bilateral_filter        cpu_multi              84.2 ms      3.7x
bilateral_filter        metal                   6.1 ms     51.2x

histogram_eq            cpu_single             15.1 ms      1.0x
histogram_eq            cpu_multi               5.8 ms      2.6x
histogram_eq            metal                   2.3 ms      6.6x

kuwahara                cpu_single            428.7 ms      1.0x
kuwahara                cpu_multi             112.3 ms      3.8x
kuwahara                metal                   5.9 ms     72.7x
```

---

## Adding a New Filter

1. Create `src/filters/MyFilter.hpp` and `src/filters/MyFilter.cpp`.
2. Inherit from `IFilter`, implement `name()`, `apply()`, and optionally `apply_parallel()`.
3. Add the `.cpp` to `FILTER_SOURCES` in `CMakeLists.txt`.
4. Register it in `src/main.cpp`:
   ```cpp
   filters.add(std::make_unique<MyFilter>());
   ```
5. Add a Metal kernel named after `MyFilter::name()` in `MetalBackend.mm` and a CUDA kernel + launcher in `filters.cu` / `CUDABackend.cpp` if GPU support is wanted.

The CPU backends work automatically with no changes.

---

## Adding a New Backend

1. Create `src/backends/MyBackend.hpp` and `.cpp`.
2. Inherit from `IBackend`, implement `name()`, `available()`, and `run()`.
3. Add the `.cpp` to `BACKEND_SOURCES` in `CMakeLists.txt` (inside a platform guard if needed).
4. Register it in `src/main.cpp`:
   ```cpp
   backends.add(std::make_unique<MyBackend>());
   ```

All filters work with the new backend automatically as long as `run()` calls `filter.apply()` or dispatches equivalent logic (i.e. the `IFilter` interface methods).

---

## Dependencies

| Dependency | Version | How obtained | Purpose |
|---|---|---|---|
| stb_image | latest (header-only) | `third_party/stb/` | PNG/JPEG load |
| stb_image_write | latest (header-only) | `third_party/stb/` | PNG/JPEG save |
| OpenMP | system | `find_package(OpenMP)` | CPU multi-threading |
| Metal | macOS system framework | CMake `enable_language(OBJCXX)` | GPU on Apple Silicon |
| CUDA Runtime | ≥ 11 | `CMAKE_CUDA_COMPILER` | GPU on NVIDIA |
| Catch2 | v3.5.3 | FetchContent (auto-downloaded) | Test framework |

No other runtime dependencies. The project builds with a standard C++17 compiler.

---

## Future Roadmap

### Planned backends

| Backend | Technology | Target |
|---|---|---|
| AMD GPU | HIP / ROCm | Windows with AMD discrete GPU |
| WebGPU | Dawn or wgpu (C bindings) | Cross-platform (browser and native) |
| WASM | Emscripten + WASM SIMD | In-browser filter preview without a server |

### Planned filters

- **Unsharp Mask** — sharpening via blurred residual
- **Median Filter** — impulse noise removal (sort-based, GPU-friendly via bitonic sort)
- **Box Blur** — O(1) per pixel using integral images; good for GPU parallelism comparison
- **Color Grading / LUT** — apply a 3D color lookup table (film emulation, tone mapping)

### Planned features

- **GUI** — a simple native window showing before/after comparison with benchmarks
- **Batch mode** — process a directory of images
- **Streaming / tiled processing** — for images larger than GPU memory
- **JSON benchmark output** — `--output-format json` for scripted comparisons
