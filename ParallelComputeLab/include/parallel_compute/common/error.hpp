#pragma once

#include <exception>
#include <string>
#include <string_view>

namespace parallel_compute {

class ParallelComputeError : public std::exception {
public:
    explicit ParallelComputeError(std::string message) : message_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

class DimensionMismatchError : public ParallelComputeError {
public:
    explicit DimensionMismatchError(std::string message)
        : ParallelComputeError(std::move(message)) {}
};

class InvalidArgumentError : public ParallelComputeError {
public:
    explicit InvalidArgumentError(std::string message)
        : ParallelComputeError(std::move(message)) {}
};

class UnsupportedHardwareError : public ParallelComputeError {
public:
    explicit UnsupportedHardwareError(std::string message)
        : ParallelComputeError(std::move(message)) {}
};

} // namespace parallel_compute
