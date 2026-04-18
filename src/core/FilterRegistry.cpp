#include "core/FilterRegistry.hpp"
#include <stdexcept>

void FilterRegistry::add(std::unique_ptr<IFilter> filter) {
    std::string n = filter->name();
    if (filters_.count(n))
        throw std::invalid_argument("FilterRegistry: duplicate filter name '" + n + "'");
    order_.push_back(n);
    filters_[n] = std::move(filter);
}

IFilter* FilterRegistry::get(const std::string& name) const {
    auto it = filters_.find(name);
    return it == filters_.end() ? nullptr : it->second.get();
}

std::vector<std::string> FilterRegistry::names() const { return order_; }
