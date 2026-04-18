cmake_minimum_required(VERSION 3.20)
project(pixel_forge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Catch2 via FetchContent
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# OpenMP
# Apple Clang does not bundle OpenMP. On macOS, help CMake find libomp from
# Homebrew (brew install libomp) before falling back to the generic search.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS
            "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP COMPONENTS CXX)

# Platform detection
include(CheckLanguage)
set(USE_METAL OFF)
set(USE_CUDA  OFF)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(USE_METAL ON)
    enable_language(OBJCXX)
    message(STATUS "[pixel_forge] Metal backend: ON")
else()
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(USE_CUDA ON)
        enable_language(CUDA)
        message(STATUS "[pixel_forge] CUDA backend: ON")
    else()
        message(STATUS "[pixel_forge] CPU-only build")
    endif()
endif()

# Shared source lists
set(CORE_SOURCES
    src/core/Image.cpp
    src/core/stb_impl.cpp       # <-- add this line
    src/core/FilterRegistry.cpp
    src/core/BackendRegistry.cpp
    src/core/Benchmark.cpp
)
set(FILTER_SOURCES
    src/filters/Grayscale.cpp
    src/filters/GaussianBlur.cpp
    src/filters/SobelEdge.cpp
    src/filters/BilateralFilter.cpp
    src/filters/HistogramEqualization.cpp
    src/filters/KuwaharaFilter.cpp
)
set(BACKEND_SOURCES
    src/backends/CPUSingleBackend.cpp
    src/backends/CPUMultiBackend.cpp
)
if(USE_METAL)
    list(APPEND BACKEND_SOURCES src/backends/MetalBackend.mm)
endif()
if(USE_CUDA)
    list(APPEND BACKEND_SOURCES src/backends/CUDABackend.cpp)
    list(APPEND BACKEND_SOURCES src/shaders/filters.cu)
endif()

# Main executable
add_executable(pixel_forge
    src/main.cpp
    ${CORE_SOURCES} ${FILTER_SOURCES} ${BACKEND_SOURCES}
)
target_include_directories(pixel_forge PRIVATE src/ third_party/)
target_compile_definitions(pixel_forge PRIVATE
    $<$<BOOL:${USE_METAL}>:USE_METAL>
    $<$<BOOL:${USE_CUDA}>:USE_CUDA>
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pixel_forge PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(pixel_forge PRIVATE USE_OPENMP)
endif()
if(USE_METAL)
    target_link_libraries(pixel_forge PRIVATE
        "-framework Metal" "-framework Foundation" "-framework QuartzCore")
endif()
if(USE_CUDA)
    set_property(TARGET pixel_forge PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Tests
add_executable(tests
    tests/test_image.cpp
    tests/test_filters.cpp
    tests/test_backends.cpp
    tests/test_registries.cpp
    tests/test_integration.cpp
    ${CORE_SOURCES} ${FILTER_SOURCES} ${BACKEND_SOURCES}
)
target_include_directories(tests PRIVATE src/ third_party/)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(tests PRIVATE
    $<$<BOOL:${USE_METAL}>:USE_METAL>
    $<$<BOOL:${USE_CUDA}>:USE_CUDA>
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tests PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(tests PRIVATE USE_OPENMP)
endif()
if(USE_METAL)
    target_link_libraries(tests PRIVATE
        "-framework Metal" "-framework Foundation" "-framework QuartzCore")
endif()
if(USE_CUDA)
    set_property(TARGET tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(tests)
