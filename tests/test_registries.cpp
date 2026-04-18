#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "core/FilterRegistry.hpp"
#include "core/BackendRegistry.hpp"
#include "core/IFilter.hpp"
#include "core/IBackend.hpp"

// Minimal stub filter
struct StubFilter : IFilter {
    std::string name() const override { return "stub"; }
    void apply(const uint8_t*, uint8_t*, int, int, int) const override {}
};

// Minimal stub backend
struct StubBackend : IBackend {
    std::string name() const override { return "stub_backend"; }
    bool available() const override { return true; }
    BenchmarkResult run(const IFilter&, const Image&) const override { return {}; }
};

TEST_CASE("FilterRegistry: register and retrieve") {
    FilterRegistry reg;
    reg.add(std::make_unique<StubFilter>());
    CHECK(reg.get("stub") != nullptr);
    CHECK(reg.get("missing") == nullptr);
    CHECK(reg.names().size() == 1);
    CHECK(reg.names()[0] == "stub");
}

TEST_CASE("BackendRegistry: register and list available") {
    BackendRegistry reg;
    reg.add(std::make_unique<StubBackend>());
    CHECK(reg.get("stub_backend") != nullptr);
    CHECK(reg.available().size() == 1);
    CHECK(reg.names()[0] == "stub_backend");
}

struct UnavailableBackend : IBackend {
    std::string name() const override { return "unavailable"; }
    bool available() const override { return false; }
    BenchmarkResult run(const IFilter&, const Image&) const override { return {}; }
};

TEST_CASE("BackendRegistry: unavailable backends excluded from available()") {
    BackendRegistry reg;
    reg.add(std::make_unique<StubBackend>());
    reg.add(std::make_unique<UnavailableBackend>());
    CHECK(reg.names().size() == 2);
    CHECK(reg.available().size() == 1);
    CHECK(reg.available()[0]->name() == "stub_backend");
}

TEST_CASE("FilterRegistry: duplicate name throws") {
    FilterRegistry reg;
    reg.add(std::make_unique<StubFilter>());
    CHECK_THROWS_AS(reg.add(std::make_unique<StubFilter>()), std::invalid_argument);
}
