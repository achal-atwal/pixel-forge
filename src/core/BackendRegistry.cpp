#include "core/BackendRegistry.hpp"
#include <stdexcept>

void BackendRegistry::add(std::unique_ptr<IBackend> backend) {
    std::string n = backend->name();
    if (backends_.count(n))
        throw std::invalid_argument("BackendRegistry: duplicate backend name '" + n + "'");
    order_.push_back(n);
    backends_[n] = std::move(backend);
}

IBackend* BackendRegistry::get(const std::string& name) const {
    auto it = backends_.find(name);
    return it == backends_.end() ? nullptr : it->second.get();
}

std::vector<IBackend*> BackendRegistry::available() const {
    std::vector<IBackend*> result;
    for (auto& n : order_) {
        auto it = backends_.find(n);
        if (it != backends_.end() && it->second->available())
            result.push_back(it->second.get());
    }
    return result;
}

std::vector<std::string> BackendRegistry::names() const { return order_; }
