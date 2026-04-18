#pragma once
#include "IFilter.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

class FilterRegistry {
public:
    FilterRegistry() = default;
    void add(std::unique_ptr<IFilter> filter);
    IFilter* get(const std::string& name) const;
    std::vector<std::string> names() const;   // insertion-ordered

private:
    std::vector<std::string>                        order_;
    std::map<std::string, std::unique_ptr<IFilter>> filters_;
};
