#pragma once
#include "IBackend.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

class BackendRegistry {
public:
    BackendRegistry() = default;
    void add(std::unique_ptr<IBackend> backend);
    IBackend* get(const std::string& name) const;
    std::vector<IBackend*>   available() const;  // only backends where available()==true
    std::vector<std::string> names()     const;  // all registered, insertion-ordered

private:
    std::vector<std::string>                         order_;
    std::map<std::string, std::unique_ptr<IBackend>> backends_;
};
